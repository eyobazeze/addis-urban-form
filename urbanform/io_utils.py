"""Loading and cleaning of boundary, building footprints and street network."""
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

SUBCITY_CANDIDATES = ["subcity", "SUBCITY", "Subcity", "ADM3_EN", "NAME_3", "name", "NAME"]


def load_boundary(path, crs, subcity_col=None):
    gdf = gpd.read_file(path).to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if subcity_col is None:
        subcity_col = next((c for c in SUBCITY_CANDIDATES if c in gdf.columns), None)
    if subcity_col is None or subcity_col not in gdf.columns:
        print(f"[boundary] no subcity name column found (columns: {list(gdf.columns)}); "
              "using generic labels. Set --subcity-col to fix.")
        gdf["subcity"] = [f"area_{i}" for i in range(len(gdf))]
    else:
        gdf = gdf.rename(columns={subcity_col: "subcity"})
    gdf["subcity"] = gdf["subcity"].astype(str)
    return gdf[["subcity", "geometry"]].reset_index(drop=True)


def _read_one(path, crs, boundary, layer, buffer_m=500.0):
    """Read one building file. Vector files are read through a bounding-box filter so a
    nationwide file is never fully loaded into memory."""
    import pyogrio
    from shapely.geometry import box

    p = Path(path)
    name = p.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(p)
        return gpd.GeoDataFrame(
            df.drop(columns="geometry"),
            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
            crs="EPSG:4326",  # Open Buildings CSVs are WGS84
        )
    if name.endswith(".parquet"):
        return gpd.read_parquet(p)

    src = str(p)
    if name.endswith(".zip"):
        src = f"/vsizip/{src}"  # GDAL reads zipped geodatabases / shapefiles in place
    if layer is None:
        layers = pyogrio.list_layers(src)
        polys = [l for l, g in layers if g and "polygon" in str(g).lower()]
        layer = polys[0] if polys else layers[0][0]
        print(f"[buildings] {p.name}: layers={[l for l, _ in layers]}, using '{layer}'")
    file_crs = pyogrio.read_info(src, layer=layer)["crs"] or "EPSG:4326"
    minx, miny, maxx, maxy = boundary.total_bounds
    bb = gpd.GeoSeries([box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)],
                       crs=boundary.crs).to_crs(file_crs).total_bounds
    gdf = pyogrio.read_dataframe(src, layer=layer, bbox=tuple(bb))
    if gdf.crs is None:
        gdf = gdf.set_crs(file_crs)
    print(f"[buildings] {p.name}: {len(gdf)} features inside the study-area bounding box")
    return gdf


def load_buildings(paths, crs, boundary, min_confidence, min_area, layer=None):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    parts = []
    for path in paths:
        g = _read_one(path, crs, boundary, layer)
        if g.crs is None:
            print(f"[buildings] {path} has no CRS; assuming EPSG:4326")
            g = g.set_crs("EPSG:4326")
        parts.append(g.to_crs(crs))
    gdf = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

    log = {"raw_in_bbox": len(gdf)}
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if "confidence" in gdf.columns:
        gdf = gdf[gdf["confidence"] >= min_confidence]
    else:
        print("[buildings] no 'confidence' column found; confidence filter skipped")
    log["after_confidence"] = len(gdf)

    gdf = gdf.copy()
    gdf["geometry"] = shapely.make_valid(gdf.geometry.to_numpy())
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"]
    gdf = gdf[gdf.geometry.area >= min_area]
    log["after_area"] = len(gdf)

    if len(parts) > 1:  # drop footprints duplicated across split files
        gdf = gdf[~gdf.geometry.to_wkb().duplicated()]
    log["after_dedup"] = len(gdf)

    city = boundary.union_all()
    shapely.prepare(city)
    cen = shapely.get_coordinates(shapely.centroid(gdf.geometry.to_numpy()))
    inside = shapely.contains_xy(city, cen[:, 0], cen[:, 1])
    gdf = gdf[inside]
    log["inside_boundary"] = len(gdf)
    print(f"[buildings] {log}")
    return gdf[["geometry"]].reset_index(drop=True), log


def load_streets(path, boundary, crs, cache_dir):
    """Return (nodes, edges) GeoDataFrames of an undirected, projected street graph."""
    if path:
        G = ox.load_graphml(path)
    else:
        poly = boundary.to_crs("EPSG:4326").union_all()
        print("[streets] downloading from OpenStreetMap (Overpass)...")
        G = ox.graph_from_polygon(poly, network_type="all", simplify=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, Path(cache_dir) / "streets.graphml")
    G = ox.project_graph(G, to_crs=crs)
    nx.set_node_attributes(G, ox.stats.count_streets_per_node(G), "street_count")
    G = ox.convert.to_undirected(G)
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"[streets] {len(nodes)} nodes, {len(edges)} undirected edges")
    return nodes, edges
