"""Generate a SYNTHETIC test city (not Addis Ababa) to verify the pipeline runs and separates known zones.

Layout (6 km x 6 km, EPSG:32637 coordinates):
  west  (x 0-2 km): planned grid  - regular 100 m street grid, regular rectangular buildings
  centre(x 2-4 km): organic dense - few streets, many small irregular buildings (the 'gap' signal)
  east  (x 4-6 km): sparse edge   - very few buildings and roads
  plus an INDUSTRIAL block in the east (x 4.75-5.5 km, y 0.5-1.25 km): a few large buildings per cell,
  high building coverage but low building count, one long road and no intersections
"""
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import shapely
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point, box

X0, Y0 = 470000.0, 990000.0
CRS = "EPSG:32637"


def _rects(cx, cy, w, h, theta):
    c, s = np.cos(theta), np.sin(theta)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    x = corners[None, :, 0] * (w[:, None] / 2)
    y = corners[None, :, 1] * (h[:, None] / 2)
    xr = cx[:, None] + x * c[:, None] - y * s[:, None]
    yr = cy[:, None] + x * s[:, None] + y * c[:, None]
    return shapely.polygons(np.stack([xr, yr], axis=-1))


def make(out_dir, seed=1):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- buildings ------------------------------------------------------
    # west: planned. 100 m blocks; 6 aligned 12x14 m buildings per block
    bx, by = np.meshgrid(np.arange(0, 2000, 100) + 50, np.arange(0, 6000, 100) + 50)
    bx, by = bx.ravel(), by.ravel()
    offs = np.array([[-25, -25], [0, -25], [25, -25], [-25, 25], [0, 25], [25, 25]])
    cx = (bx[:, None] + offs[None, :, 0]).ravel() + X0
    cy = (by[:, None] + offs[None, :, 1]).ravel() + Y0
    west = _rects(cx, cy, np.full(len(cx), 12.0), np.full(len(cx), 14.0), np.zeros(len(cx)))

    # centre: organic. ~20% coverage with small randomly rotated buildings
    n = 55000
    cx = rng.uniform(2000, 4000, n) + X0
    cy = rng.uniform(0, 6000, n) + Y0
    w = rng.uniform(4, 9, n)
    h = rng.uniform(4, 9, n)
    centre = _rects(cx, cy, w, h, rng.uniform(0, np.pi, n))

    # east: sparse, larger scattered buildings
    n = 400
    cx = rng.uniform(4000, 6000, n) + X0
    cy = rng.uniform(0, 6000, n) + Y0
    east = _rects(cx, cy, rng.uniform(8, 15, n), rng.uniform(8, 15, n), rng.uniform(0, np.pi, n))

    ind_x, ind_y = [], []
    for ci in range(19, 22):          # cell columns covering x 4750-5500
        for cj in range(2, 5):        # cell rows covering y 500-1250
            for i in range(4):
                for j in range(3):
                    ind_x.append(ci * 250 + 30 + 60 * i)
                    ind_y.append(cj * 250 + 40 + 80 * j)
    ind_x, ind_y = np.array(ind_x, dtype=float) + X0, np.array(ind_y, dtype=float) + Y0
    industrial = _rects(ind_x, ind_y, np.full(len(ind_x), 45.0), np.full(len(ind_x), 35.0), np.zeros(len(ind_x)))

    geoms = np.concatenate([west, centre, east, industrial])
    blds = gpd.GeoDataFrame({"confidence": rng.uniform(0.7, 1.0, len(geoms))}, geometry=geoms, crs=CRS)
    blds.to_file(out_dir / "buildings.gpkg", driver="GPKG")

    # ---- streets --------------------------------------------------------
    nodes, edges = [], []
    nid = 0

    def add_node(x, y):
        nonlocal nid
        nodes.append((nid, x, y))
        nid += 1
        return nid - 1

    # west: full 100 m lattice
    lattice = {}
    for i in range(0, 21):
        for j in range(0, 61):
            lattice[(i, j)] = add_node(X0 + i * 100.0, Y0 + j * 100.0)
    for (i, j), a in lattice.items():
        for di, dj in ((1, 0), (0, 1)):
            b = lattice.get((i + di, j + dj))
            if b is not None:
                edges.append((a, b))

    # centre: Delaunay on random points, keep short edges, drop ~55%
    pts = np.column_stack([rng.uniform(2000, 4000, 300), rng.uniform(0, 6000, 300)])
    ids = [add_node(X0 + x, Y0 + y) for x, y in pts]
    for simplex in Delaunay(pts).simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            p, q = pts[simplex[a]], pts[simplex[b]]
            if np.linalg.norm(p - q) < 350 and rng.random() > 0.55:
                edges.append((ids[simplex[a]], ids[simplex[b]]))

    # east: sparse long roads
    pts = np.column_stack([rng.uniform(4000, 6000, 20), rng.uniform(0, 6000, 20)])
    ids = [add_node(X0 + x, Y0 + y) for x, y in pts]
    for a in range(len(ids) - 1):
        edges.append((ids[a], ids[a + 1]))

    a_ind = add_node(X0 + 4700.0, Y0 + 900.0)
    b_ind = add_node(X0 + 5600.0, Y0 + 900.0)
    edges.append((a_ind, b_ind))

    G = nx.MultiDiGraph(crs=CRS)
    pos = {i: (x, y) for i, x, y in nodes}
    for i, (x, y) in pos.items():
        G.add_node(i, x=x, y=y)
    for a, b in set(tuple(sorted(e)) for e in edges):
        geom = LineString([pos[a], pos[b]])
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, 0, length=geom.length, highway="residential", osmid=int(a) * 100000 + int(b),
                       geometry=geom if u == a else LineString([pos[b], pos[a]]))
    nx.set_node_attributes(G, ox.stats.count_streets_per_node(G), "street_count")
    ox.save_graphml(G, out_dir / "streets.graphml")

    # ---- boundary (4 'subcities') --------------------------------------
    polys = {
        "West planned": box(X0, Y0, X0 + 2000, Y0 + 6000),
        "Centre south": box(X0 + 2000, Y0, X0 + 4000, Y0 + 3000),
        "Centre north": box(X0 + 2000, Y0 + 3000, X0 + 4000, Y0 + 6000),
        "East sparse": box(X0 + 4000, Y0, X0 + 6000, Y0 + 6000),
    }
    gpd.GeoDataFrame({"subcity": list(polys)}, geometry=list(polys.values()), crs=CRS).to_file(
        out_dir / "boundary.gpkg", driver="GPKG")
    print(f"synthetic city written to {out_dir} ({len(blds)} buildings, {G.number_of_nodes()} nodes)")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
