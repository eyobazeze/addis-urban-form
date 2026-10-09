"""Tests on a SYNTHETIC city (not Addis Ababa). The city has a planned grid (west), an organic dense area
(centre), a sparse edge and an industrial block of a few large buildings (east)."""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from make_synthetic_city import make  # noqa: E402
from urbanform.config import Config  # noqa: E402
from urbanform.io_utils import load_boundary, load_buildings, load_streets  # noqa: E402
from urbanform.metrics import cell_table, focal_sum  # noqa: E402
from urbanform.pipeline import run  # noqa: E402


def _zones(grid):
    x, y = grid.geometry.centroid.x - 470000, grid.geometry.centroid.y - 990000
    z = pd.Series(pd.cut(x, [-1, 2000, 4000, 7000], labels=["west", "centre", "east"]).astype(str), index=grid.index)
    z[(x > 4750) & (x < 5500) & (y > 500) & (y < 1250)] = "industrial"
    return z


def test_focal_sum():
    g = pd.DataFrame({"ix": [0, 1, 2, 0, 1, 2], "iy": [0, 0, 0, 1, 1, 1], "v": [1, 2, 3, 4, 5, 6]})
    out = focal_sum(g, "v", 1)
    assert list(out) == [12, 21, 16, 12, 21, 16]       # 3x3 window sums, edges truncated
    assert list(focal_sum(g, "v", 0)) == [1, 2, 3, 4, 5, 6]


def test_synthetic_city(tmp_path):
    make(tmp_path / "syn")
    cfg = Config(
        buildings_path=str(tmp_path / "syn" / "buildings.gpkg"),
        boundary_path=str(tmp_path / "syn" / "boundary.gpkg"),
        streets_path=str(tmp_path / "syn" / "streets.graphml"),
        out_dir=str(tmp_path / "out"), permutations=99, sensitivity_cell_sizes=(250.0, 500.0),
    )
    grid, log = run(cfg)
    z = _zones(grid)
    west, centre, east, ind = (grid[z == k] for k in ("west", "centre", "east", "industrial"))

    assert west["type"].nunique() == 1 and (west["type"] > 0).all()   # planned grid = one type
    assert west["dense_poorly_connected"].sum() == 0
    assert (east["built"].mean() < 0.15)                               # sparse edge mostly not clustered
    assert centre["dense_poorly_connected"].sum() > 20                 # organic dense core flagged
    assert not set(west["type"]) & set(centre["type"])                 # no type shared across zones
    assert log["validation"]["morans_I_gap"]["p_sim"] <= 0.05

    # industrial block: built, high coverage, few buildings -> must NOT be flagged on building count
    assert len(ind) == 9 and ind["built"].all()
    assert (ind["coverage"] > 0.25).all()
    assert (ind["bld_density_km2"] < log["gap_thresholds"]["density_q"] / 10).all()   # far below the "dense" cutoff
    assert ind["dense_poorly_connected"].sum() == 0

    # every flagged cell has mapped streets nearby and a class label
    fl = grid[grid["dense_poorly_connected"]]
    assert (fl["focal_segments"] > 0).all() and fl["flag_class"].isin(["typical_or_small_buildings", "large_buildings"]).all()
    for f in ["cells.gpkg", "typology_map.png", "gap_map.png", "run_log.json", "subcity_summary.csv", "flagged_cells.csv"]:
        assert (tmp_path / "out" / f).exists()

    # street access: planned grid has every building within ~25 m of a street; log carries city-wide figures
    assert west["share_far_from_street"].dropna().max() < 0.05
    assert centre["share_far_from_street"].median() > west["share_far_from_street"].median()
    assert 0 <= log["street_access"]["share_buildings_far_citywide"] <= 1
    assert "share_far_from_street" in pd.read_csv(tmp_path / "out" / "flagged_cells.csv").columns

    # with the OLD coverage-based definition, the same industrial block WOULD be flagged -> documents why count is used
    bd = load_boundary(cfg.boundary_path, cfg.crs)
    bl, _ = load_buildings(cfg.buildings_path, cfg.crs, bd, 0.7, 10.0)
    nodes, edges = load_streets(cfg.streets_path, bd, cfg.crs, tmp_path)
    cfg2 = Config(**{**cfg.__dict__, "gap_density_col": "coverage"})
    g2 = cell_table(bl, nodes, edges, bd, 250.0, cfg2)
    assert g2[_zones(g2) == "industrial"]["dense_poorly_connected"].sum() > 0


def test_street_access_known_geometry():
    import geopandas as gpd
    from shapely.geometry import LineString, Point, box
    from urbanform.metrics import add_street_access
    grid = gpd.GeoDataFrame({"cell_id": [0]}, geometry=[box(0, 0, 250, 250)], crs="EPSG:32637")
    edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (250, 0)])], crs="EPSG:32637")
    # distances to the street y=0: 10, 40, 60, 100 -> 2 of 4 farther than 50 m
    b = gpd.GeoDataFrame(geometry=[Point(50, d).buffer(2) for d in (10, 40, 60, 100)], crs="EPSG:32637")
    out = add_street_access(grid, b, edges, 50.0)
    assert out["share_far_from_street"].iloc[0] == 0.5
    assert abs(out["median_dist_street_m"].iloc[0] - 50.0) < 0.5
