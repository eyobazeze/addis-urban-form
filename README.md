# Urban Form Typology of Addis Ababa

**Question:** where has building density outpaced street-network provision, and what distinct urban form types exist?

Building footprints + OSM street network are summarised on a regular grid (default 250 m), 12 morphometric features per cell are clustered (k-means) into urban form types, and a *dense-but-poorly-connected* flag marks cells that are built up but have few intersections.

## Status
The pipeline has been run on real Addis Ababa data (Google building footprints, May 2023; OpenStreetMap streets downloaded 6 Oct 2026). Results and limits are in [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md). The code is also tested on a synthetic city (`tests/make_synthetic_city.py`, three known zones plus an industrial block); that data is not Addis Ababa and its outputs are not findings.

## Run on real data
1. `conda env create -f environment.yml && conda activate urban-form` (or add the packages to your `urban-analytics` env; install `momepy` last and check geopandas/shapely versions).
2. Download inputs into `data/raw/`:
   - **Buildings:** HDX "Ethiopia: Google Building Footprints" (two File Geodatabase zips, `eth_buildings_part1.gdb.zip` and `part2`). Pass both files to `--buildings`; the loader reads only the Addis Ababa bounding box from each (it never loads the whole country), merges them and removes duplicates. If a zip holds several layers, pick one with `--layer`. Other sources (Open Buildings CSV.GZ, GeoParquet, GPKG, GeoJSON, shapefile) also work. Column names in the HDX files are unverified; the confidence filter is skipped automatically if there is no `confidence` column.
   - **Boundary:** subcity polygons (11 subcities). Confirm the source and licence yourself; set `--subcity-col` to the name column.
   - **Streets:** leave `--streets` empty to download via OSMnx, or pass a saved `.graphml`.
3. `python run_pipeline.py --buildings data/raw/eth_buildings_part1.gdb --boundary "data/raw/Ethiopia - Addis Ababa Subcities/Ethiopia - Addis Ababa Subcities.shp" --subcity-col Sub_City --k-min 2 --out outputs/addis` (Part 1 holds all Addis footprints; Part 2 has none inside the city.)
4. Test: `pytest -q tests`

Useful flags: `--cell-size`, `--k` (fix number of types), `--k-min/--k-max`, `--min-buildings`, `--min-confidence`.

## Outputs (`outputs/<run>/`)
`cells.gpkg` (all metrics, type, gap flag) · `typology_map.png` · `gap_map.png` · `lisa_gap_map.png` · `type_profiles_heatmap.png` · `type_profiles_raw.csv` / `_zscores.csv` · `flagged_cells.csv` (now with `median_dist_street_m`, `share_far_from_street`) · `subcity_type_mix.png` · `subcity_summary.csv` · `sensitivity_gap_by_subcity.csv` · `k_selection.csv` · `run_log.json` (config, cleaning counts, dropped features, validation statistics).

## Method notes
- **Features:** building coverage, density, mean footprint area, area CV, compactness, elongation, nearest-neighbour spacing; street intersection density, street length density, dead-end share, circuity, mean segment length. Features correlated above |r| = 0.90 are dropped (listed in `run_log.json`).
- **Not clustered:** cells with fewer than 5 buildings are "sparse / unbuilt".
- **Gap metric (v2):** among built cells, `gap = percentile(buildings per km2) - percentile(neighbourhood-averaged intersection density)`. Connectivity is averaged over a ~750 m window (`focal_width_m`) because a single 250 m cell holds only 0-3 intersections. Flag = building density in the top quartile *and* connectivity in the bottom quartile, and only where at least one street segment is mapped inside the window (otherwise it is missing data, counted in `run_log.json` as `dense_cells_excluded_for_no_street_data`). Density is counted in buildings, not covered area, so a few very large buildings (industrial, commercial) are not read as dense housing. Flagged cells are labelled `typical_or_small_buildings` or `large_buildings` by their median building area; `flagged_cells.csv` lists each flagged cell with latitude/longitude for checking on satellite imagery. Thresholds are relative (quartiles within this city), so the flag always finds the relative extremes; it does not say any area is inadequate in absolute terms.
- **Street access (v3):** for every building, distance from its centroid to the nearest mapped street segment. Per cell: `share_far_from_street` (share of buildings farther than `Config.street_access_m`, default 50 m; no CLI flag) and `median_dist_street_m`. `run_log.json` has `street_access` with the building-weighted share citywide and inside flagged cells. Not a clustering feature. It cannot tell a real lack of streets from unmapped lanes, so read it together with the satellite check.
- **Validation:** Moran's I on gap and coverage, LISA clusters, adjacency coherence of types vs a permutation null, k-means seed stability (ARI), subcity rank correlation of the flag across 150/250/500 m grids.

## Known limitations (state these in any write-up)
- **Choosing k:** the silhouette score favoured k = 7 on the synthetic city, splitting one homogeneous organic zone into five types. Do not accept the automatic k blindly. Compare `k_selection.csv`, the profile heatmap and imagery, and fix `--k` to the most interpretable value.
- **Footprint quality:** in dense informal areas, Open Buildings can merge adjacent roofs or miss buildings. This biases coverage, area and spacing. Check a manual sample against satellite imagery.
- **OSM completeness:** unmapped informal streets make cells look "poorly connected" when they are not. The gap flag mixes real under-provision with mapping gaps; cells with zero mapped segments are counted in `run_log.json`.
- **Grid (MAUP):** results depend on cell size and placement; the sensitivity table shows how stable the subcity-level flag is.
- **Unlabeled types:** the types are statistical clusters. Names such as "planned grid" or "dense organic" must come from your reading of profiles and imagery.
