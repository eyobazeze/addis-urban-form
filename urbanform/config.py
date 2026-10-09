"""Run configuration. Every threshold used in the analysis lives here so it is easy to report."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- inputs ---------------------------------------------------------
    buildings_path: str | list = ""     # one path or a list: CSV/CSV.GZ, Parquet, GPKG, GeoJSON, .gdb(.zip), shp(.zip)
    buildings_layer: str | None = None  # layer name inside a GDB/GPKG (auto-picks first polygon layer)
    boundary_path: str = ""             # subcity polygons (GPKG/GeoJSON/SHP), any CRS
    streets_path: str | None = None     # .graphml; None = download from OSM via OSMnx
    out_dir: str = "outputs"
    crs: str = "EPSG:32637"             # UTM 37N, metric CRS for Addis Ababa
    subcity_col: str | None = None      # name column in boundary file (auto-detected if None)

    # --- spatial unit ---------------------------------------------------
    cell_size: float = 250.0            # metres
    min_land_frac: float = 0.5          # keep grid cells with >= this share inside the boundary

    # --- building cleaning ---------------------------------------------
    min_confidence: float = 0.70        # Open Buildings confidence filter (if column exists)
    min_building_area: float = 10.0     # m2

    # --- clustering -----------------------------------------------------
    min_buildings_per_cell: int = 5     # cells below this are "sparse / unbuilt", not clustered
    k: int | None = None                # None = choose by silhouette within [k_min, k_max]
    k_min: int = 4
    k_max: int = 8
    corr_threshold: float = 0.90        # drop a feature if |r| with an earlier one exceeds this
    z_clip: float = 4.0                 # clip standardised features to +/- this
    seed: int = 42

    # --- gap metric -----------------------------------------------------
    gap_dense_q: float = 0.75           # "dense" = building coverage >= this quantile of built cells
    gap_poor_q: float = 0.25            # "poorly connected" = intersection density <= this quantile
    gap_density_col: str = "bld_density_km2"   # "dense" measure: buildings per km2 (or "coverage")
    focal_width_m: float = 750.0        # connectivity is averaged over a window this wide (cells around each cell)
    large_footprint_q: float = 0.75     # flagged cells whose median building area exceeds this quantile of built cells -> "large_buildings"

    street_access_m: float = 50.0       # a building is "far" if its centroid is farther than this from any mapped street

    # --- validation -----------------------------------------------------
    permutations: int = 999
    sensitivity_cell_sizes: tuple = (150.0, 250.0, 500.0)
