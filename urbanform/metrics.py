"""Grid construction and per-cell building / street metrics."""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import box


def make_grid(boundary, cell_size, min_land_frac):
    minx, miny, maxx, maxy = boundary.total_bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    ix, iy, cells = [], [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ix.append(i)
            iy.append(j)
            cells.append(box(x, y, x + cell_size, y + cell_size))
    grid = gpd.GeoDataFrame({"cell_id": np.arange(len(cells)), "ix": ix, "iy": iy},
                            geometry=cells, crs=boundary.crs)
    land = grid.geometry.intersection(boundary.union_all()).area
    grid["land_frac"] = land.to_numpy() / cell_size**2
    grid = grid[grid["land_frac"] >= min_land_frac].reset_index(drop=True)
    grid["cell_id"] = np.arange(len(grid))
    grid["land_area"] = grid["land_frac"] * cell_size**2
    return grid


def _building_shape(geoms):
    area = shapely.area(geoms)
    perim = shapely.length(geoms)
    compact = np.where(perim > 0, 4 * np.pi * area / np.where(perim > 0, perim, 1) ** 2, np.nan)
    rect = shapely.minimum_rotated_rectangle(geoms)
    ok = shapely.get_num_coordinates(rect) == 5
    elong = np.full(len(geoms), np.nan)
    if ok.any():
        c = shapely.get_coordinates(rect[ok]).reshape(-1, 5, 2)
        s1 = np.linalg.norm(c[:, 1] - c[:, 0], axis=1)
        s2 = np.linalg.norm(c[:, 2] - c[:, 1], axis=1)
        lo, hi = np.minimum(s1, s2), np.maximum(s1, s2)
        elong[ok] = np.where(hi > 0, lo / np.where(hi > 0, hi, 1), np.nan)
    return area, compact, elong


def add_building_metrics(grid, buildings):
    geoms = buildings.geometry.to_numpy()
    area, compact, elong = _building_shape(geoms)
    xy = shapely.get_coordinates(shapely.centroid(geoms))
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]  # distance to nearest other building centroid
    pts = gpd.GeoDataFrame(
        {"area": area, "compact": compact, "elong": elong, "nn": nn},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=buildings.crs,
    )
    j = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    g = j.groupby("cell_id")
    agg = pd.DataFrame({
        "n_buildings": g.size(),
        "built_area": g["area"].sum(),
        "mean_area": g["area"].mean(),
        "median_area": g["area"].median(),
        "cv_area": g["area"].std() / g["area"].mean(),
        "mean_compact": g["compact"].mean(),
        "mean_elong": g["elong"].mean(),
        "mean_nn": g["nn"].mean(),
    })
    out = grid.merge(agg, left_on="cell_id", right_index=True, how="left")
    out["n_buildings"] = out["n_buildings"].fillna(0).astype(int)
    out["built_area"] = out["built_area"].fillna(0.0)
    out["cv_area"] = out["cv_area"].fillna(0.0)
    out["coverage"] = (out["built_area"] / out["land_area"]).clip(upper=1.0)
    out["bld_density_km2"] = out["n_buildings"] / (out["land_area"] / 1e6)
    return out


def add_street_metrics(grid, nodes, edges):
    geoms = edges.geometry.to_numpy()
    length = shapely.length(geoms)
    straight = shapely.distance(shapely.get_point(geoms, 0), shapely.get_point(geoms, -1))
    mid = shapely.get_coordinates(shapely.line_interpolate_point(geoms, 0.5, normalized=True))
    e = gpd.GeoDataFrame(
        {"length": length, "straight": straight},
        geometry=gpd.points_from_xy(mid[:, 0], mid[:, 1]), crs=edges.crs,
    )
    je = gpd.sjoin(e, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    je = je[~je.index.duplicated()]
    ge = je.groupby("cell_id")
    agg_e = pd.DataFrame({
        "street_len": ge["length"].sum(),
        "n_segments": ge.size(),
        "straight_sum": ge["straight"].sum(),
    })
    agg_e["mean_seg_len"] = agg_e["street_len"] / agg_e["n_segments"]
    agg_e["circuity"] = agg_e["street_len"] / agg_e["straight_sum"].replace(0, np.nan)

    n = gpd.GeoDataFrame({"sc": nodes["street_count"].to_numpy()},
                         geometry=nodes.geometry.to_numpy(), crs=nodes.crs)
    jn = gpd.sjoin(n, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    jn = jn[~jn.index.duplicated()]
    gn = jn.groupby("cell_id")
    agg_n = pd.DataFrame({
        "n_intersections": gn["sc"].apply(lambda s: int((s >= 3).sum())),
        "n_dead_ends": gn["sc"].apply(lambda s: int((s == 1).sum())),
    })

    out = grid.merge(agg_e, left_on="cell_id", right_index=True, how="left")
    out = out.merge(agg_n, left_on="cell_id", right_index=True, how="left")
    for c in ["street_len", "n_segments", "n_intersections", "n_dead_ends"]:
        out[c] = out[c].fillna(0)
    km2 = out["land_area"] / 1e6
    out["intersection_density_km2"] = out["n_intersections"] / km2
    out["street_density_km_km2"] = (out["street_len"] / 1000.0) / km2
    denom = out["n_intersections"] + out["n_dead_ends"]
    out["dead_end_share"] = np.where(denom > 0, out["n_dead_ends"] / denom.replace(0, np.nan), np.nan)
    out["no_streets"] = out["n_segments"] == 0
    return out.drop(columns=["straight_sum"])


def add_street_access(grid, buildings, edges, threshold_m):
    """Per cell: share of buildings whose centroid is more than `threshold_m` from the nearest mapped street segment.

    Uses all buildings (not only built cells), so it is defined for every cell with at least one building.
    Not a clustering feature. A high share means buildings are far from any street OSM knows about; it cannot
    separate a real lack of streets from unmapped lanes.
    """
    out = grid.copy()
    out["median_dist_street_m"] = np.nan
    out["share_far_from_street"] = np.nan
    if len(edges) == 0 or len(buildings) == 0:
        return out
    pts = shapely.centroid(buildings.geometry.to_numpy())
    tree = shapely.STRtree(edges.geometry.to_numpy())
    _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    xy = shapely.get_coordinates(pts)
    p = gpd.GeoDataFrame({"d": dist}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=buildings.crs)
    j = gpd.sjoin(p, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    g = j.groupby("cell_id")["d"]
    agg = pd.DataFrame({"median_dist_street_m": g.median(), "share_far_from_street": g.apply(lambda s: float((s > threshold_m).mean()))})
    out = out.drop(columns=["median_dist_street_m", "share_far_from_street"]).merge(agg, left_on="cell_id", right_index=True, how="left")
    return out


def focal_sum(grid, col, half_width):
    """Sum of `col` over the (2*half_width+1)^2 block of cells centred on each cell."""
    from scipy.ndimage import convolve
    nx, ny = int(grid["ix"].max()) + 1, int(grid["iy"].max()) + 1
    arr = np.zeros((nx, ny))
    arr[grid["ix"].to_numpy(), grid["iy"].to_numpy()] = grid[col].to_numpy(dtype=float)
    k = np.ones((2 * half_width + 1, 2 * half_width + 1))
    out = convolve(arr, k, mode="constant", cval=0.0)
    return out[grid["ix"].to_numpy(), grid["iy"].to_numpy()]


def add_gap_metric(grid, cfg, cell_size):
    """Dense-but-poorly-connected flag, among built cells only.

    - density: buildings per km2 (cfg.gap_density_col); count-based so a few huge buildings do not look 'dense'
    - connectivity: intersections per km2 averaged over a ~cfg.focal_width_m window (single 250 m cells hold
      only 0-3 intersections, which is too coarse to rank)
    - cells with no mapped street segment anywhere in the window are excluded (missing data, not low connectivity)
    - flagged cells are labelled 'large_buildings' if their median building area is in the top
      (1 - cfg.large_footprint_q) share of built cells, otherwise 'typical_or_small_buildings'
    """
    out = grid.copy()
    w = max(0, int(round((cfg.focal_width_m / cell_size - 1) / 2)))
    land = focal_sum(out, "land_area", w)
    out["focal_land_area"] = land
    out["conn_focal_km2"] = np.where(land > 0, focal_sum(out, "n_intersections", w) / np.where(land > 0, land, 1) * 1e6, np.nan)
    out["focal_segments"] = focal_sum(out, "n_segments", w)
    built = out["n_buildings"] >= cfg.min_buildings_per_cell
    out["built"] = built
    out["gap"] = np.nan
    out["dense_poorly_connected"] = False
    out["flag_class"] = ""
    dcol = cfg.gap_density_col
    if built.sum() == 0:
        return out
    b = out.loc[built]
    out.loc[built, "gap"] = b[dcol].rank(pct=True) - b["conn_focal_km2"].rank(pct=True)
    q_dense = b[dcol].quantile(cfg.gap_dense_q)
    q_poor = b["conn_focal_km2"].quantile(cfg.gap_poor_q)
    has_streets = out["focal_segments"] > 0
    dense = built & (out[dcol] >= q_dense)
    flag = dense & has_streets & (out["conn_focal_km2"] <= q_poor)
    out["dense_poorly_connected"] = flag
    q_large = b["median_area"].quantile(cfg.large_footprint_q)
    out.loc[flag, "flag_class"] = np.where(out.loc[flag, "median_area"] > q_large, "large_buildings", "typical_or_small_buildings")
    out.attrs["gap_thresholds"] = {
        "density_col": dcol, "density_q": float(q_dense),
        "connectivity_focal_km2_q": float(q_poor), "focal_half_width_cells": w,
        "median_area_large_threshold_m2": float(q_large),
        "dense_cells_excluded_for_no_street_data": int((dense & ~has_streets).sum()),
    }
    return out


def cell_table(buildings, nodes, edges, boundary, cell_size, cfg):
    grid = make_grid(boundary, cell_size, cfg.min_land_frac)
    grid = add_building_metrics(grid, buildings)
    grid = add_street_metrics(grid, nodes, edges)
    grid = add_street_access(grid, buildings, edges, cfg.street_access_m)
    return add_gap_metric(grid, cfg, cell_size)
