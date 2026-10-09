"""End-to-end run: load -> grid metrics -> cluster -> validate -> outputs."""
import json
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import cluster, plots, validate
from .io_utils import load_boundary, load_buildings, load_streets
from .metrics import cell_table


def subcity_summary(grid, boundary):
    cen = grid[["cell_id", "built", "dense_poorly_connected", "coverage", "share_far_from_street"]].copy()
    cen = gpd.GeoDataFrame(cen, geometry=grid.geometry.centroid, crs=grid.crs)
    j = gpd.sjoin(cen, boundary[["subcity", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    j["subcity"] = j["subcity"].fillna("unassigned")
    built = j[j["built"]]
    out = pd.DataFrame({
        "n_cells": j.groupby("subcity").size(),
        "n_built_cells": built.groupby("subcity").size(),
        "n_flagged": built.groupby("subcity")["dense_poorly_connected"].sum(),
        "mean_coverage_built": built.groupby("subcity")["coverage"].mean(),
        "median_cell_share_far_from_street": built.groupby("subcity")["share_far_from_street"].median(),
    }).fillna(0)
    out["share_flagged_of_built"] = np.where(out["n_built_cells"] > 0, out["n_flagged"] / out["n_built_cells"].replace(0, np.nan), np.nan)
    return out, j.set_index("cell_id")["subcity"]


def run(cfg):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    boundary = load_boundary(cfg.boundary_path, cfg.crs, cfg.subcity_col)
    buildings, bld_log = load_buildings(cfg.buildings_path, cfg.crs, boundary,
                                        cfg.min_confidence, cfg.min_building_area, cfg.buildings_layer)
    nodes, edges = load_streets(cfg.streets_path, boundary, cfg.crs, Path(cfg.out_dir).parent / "data" / "processed")

    grid = cell_table(buildings, nodes, edges, boundary, cfg.cell_size, cfg)
    grid["cell_size"] = cfg.cell_size
    sub_tbl, cell_subcity = subcity_summary(grid, boundary)
    grid["subcity"] = grid["cell_id"].map(cell_subcity)

    built = grid[grid["built"]].copy()
    if len(built) < 50:
        raise RuntimeError(f"Only {len(built)} built cells; check inputs, CRS and --min-buildings.")

    # ---- clustering ----------------------------------------------------
    Z, kept, dropped, imputed = cluster.prepare_features(built, cfg.corr_threshold, cfg.z_clip)
    k_table = cluster.select_k(Z, cfg.k_min, cfg.k_max, cfg.seed)
    k = cfg.k or int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
    labels, _ = cluster.fit_kmeans(Z, k, cfg.seed)
    stability = cluster.seed_stability(Z, k, cfg.seed)
    grid["type"] = 0
    grid.loc[built.index, "type"] = labels + 1
    raw_prof, z_prof = cluster.profile(built, labels, kept)

    # ---- validation ----------------------------------------------------
    built = grid[grid["built"]].copy()
    validation = {
        "morans_I_gap": validate.morans(built, "gap", cfg.permutations, cfg.seed),
        "morans_I_coverage": validate.morans(built, "coverage", cfg.permutations, cfg.seed),
        "type_coherence": validate.type_coherence(built, "type", cfg.permutations, cfg.seed),
        "kmeans_seed_stability_ARI": stability,
        "k_selected": k,
        "silhouette_at_k": float(k_table.loc[k_table["k"] == k, "silhouette"].iloc[0]),
    }
    lisa_gap = validate.lisa(built, "gap", cfg.permutations, cfg.seed)
    hh = (lisa_gap["lisa_gap"] == "HH").groupby(lisa_gap["subcity"]).agg(["sum", "mean"])
    hh.columns = ["n_gap_hotspot_cells", "share_gap_hotspot_of_built"]

    # grid-size sensitivity of the dense-but-poorly-connected flag
    tables = {cfg.cell_size: grid}
    for cs in cfg.sensitivity_cell_sizes:
        if cs != cfg.cell_size:
            tables[cs] = cell_table(buildings, nodes, edges, boundary, cs, cfg)
    sens_df, sens_pairs = validate.gap_sensitivity(tables, lambda g: subcity_summary(g, boundary)[0])
    validation["gap_flag_subcity_rank_correlation_across_grid_sizes"] = sens_pairs

    # ---- type mix by subcity ------------------------------------------
    mix = (grid[grid["type"] > 0].groupby(["subcity", "type"]).size().unstack(fill_value=0))
    mix_share = mix.div(mix.sum(axis=1), axis=0)

    # ---- write ---------------------------------------------------------
    keep_cols = [c for c in grid.columns if c != "geometry"]
    grid[keep_cols + ["geometry"]].to_file(out / "cells.gpkg", layer="cells", driver="GPKG")
    raw_prof.to_csv(out / "type_profiles_raw.csv")
    z_prof.to_csv(out / "type_profiles_zscores.csv")
    k_table.to_csv(out / "k_selection.csv", index=False)
    sub_tbl.join(hh).join(mix_share.add_prefix("share_type_")).to_csv(out / "subcity_summary.csv")
    sens_df.to_csv(out / "sensitivity_gap_by_subcity.csv")

    fl = grid[grid["dense_poorly_connected"]].copy()
    if len(fl):
        ll = fl.geometry.centroid.to_crs("EPSG:4326")
        fl["lat"], fl["lon"] = ll.y.round(6).to_numpy(), ll.x.round(6).to_numpy()
        fl["lat_lon"] = fl["lat"].astype(str) + ", " + fl["lon"].astype(str)
        fl[["cell_id", "subcity", "flag_class", "n_buildings", "bld_density_km2", "median_area", "coverage",
            "conn_focal_km2", "median_dist_street_m", "share_far_from_street", "lat", "lon", "lat_lon"]].sort_values(["flag_class", "subcity"]).to_csv(
            out / "flagged_cells.csv", index=False)

    plots.typology_map(grid, boundary, out / "typology_map.png")
    plots.gap_map(grid, boundary, out / "gap_map.png")
    plots.profile_heatmap(z_prof, out / "type_profiles_heatmap.png")
    plots.subcity_types(mix_share, out / "subcity_type_mix.png")
    plots.lisa_map(lisa_gap, "gap", boundary, out / "lisa_gap_map.png")

    # city-wide building-level access (weighted by buildings, not cells)
    bsum = grid.loc[grid["share_far_from_street"].notna()]
    far_n = (bsum["share_far_from_street"] * bsum["n_buildings"]).sum()
    flg = bsum[bsum["dense_poorly_connected"]]
    street_access = {
        "threshold_m": cfg.street_access_m,
        "share_buildings_far_citywide": float(far_n / bsum["n_buildings"].sum()),
        "share_buildings_far_in_flagged_cells": float((flg["share_far_from_street"] * flg["n_buildings"]).sum() / flg["n_buildings"].sum()) if len(flg) else None,
        "n_buildings_far_citywide": int(round(far_n)),
        "n_buildings_total": int(bsum["n_buildings"].sum()),
    }
    print(json.dumps(street_access, indent=2))

    log = {
        "street_access": street_access,
        "config": asdict(cfg),
        "building_cleaning": bld_log,
        "n_cells": int(len(grid)), "n_built_cells": int(grid["built"].sum()),
        "n_dense_poorly_connected": int(grid["dense_poorly_connected"].sum()),
        "flag_classes": grid.loc[grid["dense_poorly_connected"], "flag_class"].value_counts().to_dict(),
        "gap_thresholds": grid.attrs.get("gap_thresholds"),
        "features_used": kept, "features_dropped_for_correlation": dropped,
        "features_imputed_with_median": imputed,
        "cells_with_no_street_segments": int(grid["no_streets"].sum()),
        "validation": validation,
    }
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(validation, indent=2))
    return grid, log
