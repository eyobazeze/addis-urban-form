# Where has building density outpaced street provision in Addis Ababa?

I divided Addis Ababa into 250 m cells using Google building footprints (May 2023) and the OpenStreetMap street network (downloaded 6 Oct 2026). Of 8,304 cells, 6,835 contain at least 5 buildings. Clustering 12 form metrics split these into two urban form types: a denser, better-connected core and a sparse fringe with long street segments and many dead ends.

I flagged cells that are dense (at least 3,904 buildings/km², top quartile) and poorly connected (at most 62.2 intersections/km² over a 750 m window, bottom quartile). **49 cells** qualify, mostly small buildings (median footprint 32-55 m²): Yeka 19, Kolfe Keranyo 13, Bole 9, Akaki Kality 4, Nefas Silk-Lafto 2, Gulele 1, unassigned 1.

The flagged cells are not all alike. In 32, nearly all buildings are within 50 m of a mapped street, so the issue is few intersections. In 17, at least 30% of buildings are more than 50 m from a mapped street, and in 8 (7 in Kolfe Keranyo) it is 60% or more. Across the 49 flagged cells, 24.5% of buildings are more than 50 m from a mapped street, against 5.5% citywide.

## Validation and limits
I checked 14 flagged cells on satellite imagery. In 2 (9.00056, 38.654961 and 9.000566, 38.661784), lanes exist that OpenStreetMap lacks, so part of the signal is mapping gaps. Thresholds are quartiles within Addis Ababa, so they show relative, not absolute, under-provision. Footprints may merge in dense informal areas. Subcity rankings are stable between 150 m and 250 m grids (rank correlation 0.94) but not at 500 m (0.31), and there are only 11 subcities.

Figures come from `outputs/<run>/run_log.json` and `flagged_cells.csv` of the run described in the README.
