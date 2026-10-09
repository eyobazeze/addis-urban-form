"""Command line entry point.

Example (real data):
    python run_pipeline.py --buildings data/raw/open_buildings_addis.csv.gz \
        --boundary data/raw/addis_subcities.gpkg --out outputs
"""
import argparse

from urbanform.config import Config
from urbanform.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Urban form typology from building footprints + street network")
    p.add_argument("--buildings", required=True, nargs="+", help="one or more building files (e.g. both HDX parts)")
    p.add_argument("--layer", default=None, help="layer name inside a .gdb/.gpkg")
    p.add_argument("--boundary", required=True)
    p.add_argument("--streets", default=None, help=".graphml; omit to download from OSM")
    p.add_argument("--out", default="outputs")
    p.add_argument("--crs", default="EPSG:32637")
    p.add_argument("--subcity-col", default=None)
    p.add_argument("--cell-size", type=float, default=250.0)
    p.add_argument("--k", type=int, default=None, help="fix number of types; default picks by silhouette")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--min-buildings", type=int, default=5)
    p.add_argument("--min-confidence", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = Config(
        buildings_path=a.buildings, buildings_layer=a.layer, boundary_path=a.boundary, streets_path=a.streets, out_dir=a.out,
        crs=a.crs, subcity_col=a.subcity_col, cell_size=a.cell_size, k=a.k, k_min=a.k_min, k_max=a.k_max,
        min_buildings_per_cell=a.min_buildings, min_confidence=a.min_confidence, seed=a.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
