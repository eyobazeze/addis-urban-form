"""Spatial validation: Moran's I, LISA, type coherence, and grid-size sensitivity."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _weights(sub):
    from libpysal.weights import Queen
    w = Queen.from_dataframe(sub.reset_index(drop=True), use_index=False, silence_warnings=True)
    w.transform = "r"
    return w


def morans(sub, col, permutations, seed):
    import esda
    sub = sub.dropna(subset=[col]).reset_index(drop=True)
    w = _weights(sub)
    y = sub[col].to_numpy(dtype=float)
    np.random.seed(seed)  # esda.Moran has no seed argument; it draws from numpy's global generator
    mi = esda.Moran(y, w, permutations=permutations)
    return {"I": float(mi.I), "p_sim": float(mi.p_sim), "n": int(len(sub))}


def lisa(sub, col, permutations, seed):
    """Return sub (reset index) with LISA quadrant label in 'lisa_<col>'."""
    import esda
    sub = sub.dropna(subset=[col]).reset_index(drop=True)
    w = _weights(sub)
    y = sub[col].to_numpy(dtype=float)
    lm = esda.Moran_Local(y, w, permutations=permutations, seed=seed, alternative="two-sided")
    names = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    lab = np.array([names[q] if p < 0.05 else "ns" for q, p in zip(lm.q, lm.p_sim)])
    sub[f"lisa_{col}"] = lab
    return sub


def type_coherence(sub, type_col, permutations, seed):
    """Share of adjacent cell pairs with the same type vs a label-permutation null."""
    sub = sub.reset_index(drop=True)
    w = _weights(sub)
    rows, cols = w.sparse.nonzero()
    lab = sub[type_col].to_numpy()
    obs = float(np.mean(lab[rows] == lab[cols]))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean((p := rng.permutation(lab))[rows] == p[cols]) for _ in range(permutations)])
    return {
        "observed_same_type_share": obs,
        "null_mean": float(null.mean()),
        "null_95th": float(np.quantile(null, 0.95)),
        "p_sim": float((1 + (null >= obs).sum()) / (permutations + 1)),
    }


def gap_sensitivity(tables, subcity_summary_fn):
    """tables: {cell_size: grid}. Compare subcity share of dense-but-poorly-connected cells."""
    shares = {}
    for cs, grid in tables.items():
        s = subcity_summary_fn(grid)
        shares[cs] = s["share_flagged_of_built"]
    df = pd.DataFrame(shares)
    df.columns = [f"{int(c)}m" for c in df.columns]
    pair = {}
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = spearmanr(df[cols[i]], df[cols[j]], nan_policy="omit").statistic
            pair[f"{cols[i]} vs {cols[j]}"] = None if np.isnan(rho) else float(rho)
    return df, pair
