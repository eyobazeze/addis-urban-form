"""Feature preparation and clustering of built cells into urban form types."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "coverage", "bld_density_km2", "mean_area", "cv_area", "mean_compact", "mean_elong", "mean_nn",
    "intersection_density_km2", "street_density_km_km2", "dead_end_share", "circuity", "mean_seg_len",
]
LOG_FEATURES = {"bld_density_km2", "mean_area", "mean_nn", "intersection_density_km2",
                "street_density_km_km2", "mean_seg_len"}


def prepare_features(df, corr_threshold, z_clip):
    """Return (X_scaled, kept_columns, dropped_info, imputed_counts) for a built-cell table."""
    X = df[FEATURES].copy()
    imputed = {c: int(X[c].isna().sum()) for c in FEATURES if X[c].isna().any()}
    X = X.fillna(X.median())
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    corr = X.corr().abs()
    kept, dropped = [], {}
    for c in FEATURES:
        clash = next((k for k in kept if corr.loc[c, k] > corr_threshold), None)
        if clash is None:
            kept.append(c)
        else:
            dropped[c] = f"|r|={corr.loc[c, clash]:.2f} with {clash}"
    # drop zero-variance columns
    kept = [c for c in kept if X[c].std() > 0]
    Z = StandardScaler().fit_transform(X[kept])
    Z = np.clip(Z, -z_clip, z_clip)
    return Z, kept, dropped, imputed


def select_k(Z, k_min, k_max, seed, sample_size=10000):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(sample_size, len(Z)), random_state=seed)
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def fit_kmeans(Z, k, seed):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    # relabel by cluster size (largest = 1) so type numbers are reproducible
    order = np.argsort(-np.bincount(km.labels_))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in km.labels_]), km


def seed_stability(Z, k, seed, n=10):
    """Mean pairwise adjusted Rand index of k-means solutions from different random starts."""
    labs = [KMeans(n_clusters=k, n_init=1, random_state=seed + i).fit_predict(Z) for i in range(n)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(aris))


def profile(df, labels, kept):
    d = df.copy()
    d["type"] = labels + 1
    raw = d.groupby("type")[kept].mean()
    raw.insert(0, "n_cells", d.groupby("type").size())
    raw.insert(1, "share_of_built_cells", raw["n_cells"] / len(d))
    z = (d[kept] - d[kept].mean()) / d[kept].std(ddof=0)
    z["type"] = d["type"]
    zmean = z.groupby("type")[kept].mean()
    return raw, zmean
