"""
Download and preprocess the PBMC 3k dataset for scAnimator demo.

Produces pbmc3k.h5ad with:
  - Normalized + log-transformed expression in .X
  - Full raw gene set in .raw
  - PCA in .obsm['X_pca']
  - Cell type annotations in .obs['cell_type']

Cell type annotation uses canonical PBMC markers:
  CD4 T       — IL7R, CD4
  CD8 T       — CD8A, CD8B
  NK          — GNLY, NKG7
  B           — MS4A1 (CD20)
  CD14+ Mono  — CD14, LYZ
  FCGR3A+ Mono — FCGR3A, MS4A7
  Dendritic   — FCER1A, CST3
  Platelet    — PPBP

Usage: python prepare_pbmc3k.py
Output: pbmc3k.h5ad in the current directory
"""

import scanpy as sc
import numpy as np

print("Downloading PBMC 3k dataset...")
adata = sc.datasets.pbmc3k()
print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

# ── QC and filtering ──────────────────────────────────────
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Mitochondrial QC
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 5, :].copy()
print(f"  After QC: {adata.shape[0]} cells x {adata.shape[1]} genes")

# ── Normalize ─────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store full gene set in .raw before HVG filtering
adata.raw = adata

# ── HVG + PCA ─────────────────────────────────────────────
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
print(f"  HVG: {adata.shape[1]} genes, PCA: {adata.obsm['X_pca'].shape}")

# ── Neighbors + clustering ────────────────────────────────
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)  # 2D UMAP for marker validation (not used by scAnimator)

# ── Cell type annotation via canonical markers ────────────
marker_genes = {
    "CD4 T": ["IL7R"],
    "CD8 T": ["CD8A", "CD8B"],
    "NK": ["GNLY", "NKG7"],
    "B": ["MS4A1"],
    "CD14+ Mono": ["CD14", "LYZ"],
    "FCGR3A+ Mono": ["FCGR3A", "MS4A7"],
    "Dendritic": ["FCER1A", "CST3"],
    "Platelet": ["PPBP"],
}

# Score each cell for each type using raw expression
print("Annotating cell types...")
raw_df = adata.raw.to_adata()
sc.pp.scale(raw_df, max_value=10)

cell_type = np.full(adata.n_obs, "Unknown", dtype=object)
scores = np.zeros((adata.n_obs, len(marker_genes)))

for i, (ct, markers) in enumerate(marker_genes.items()):
    available = [g for g in markers if g in raw_df.var_names]
    if available:
        idx = [list(raw_df.var_names).index(g) for g in available]
        X = raw_df.X[:, idx]
        if hasattr(X, "toarray"):
            X = X.toarray()
        scores[:, i] = X.mean(axis=1)

best = scores.argmax(axis=1)
type_names = list(marker_genes.keys())
for i in range(adata.n_obs):
    if scores[i, best[i]] > 0:
        cell_type[i] = type_names[best[i]]

adata.obs["cell_type"] = cell_type
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

# Print summary
print("  Cell type counts:")
for ct, count in adata.obs["cell_type"].value_counts().items():
    print(f"    {ct}: {count}")

# ── Save ──────────────────────────────────────────────────
out_path = "pbmc3k.h5ad"
adata.write(out_path)
print(f"\nSaved to {out_path}")
print(f"  {adata.shape[0]} cells, {adata.shape[1]} HVG genes, {adata.raw.shape[1]} total genes")
