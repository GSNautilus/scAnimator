"""
scAnimator: Build UMAP warp viewer with external data files.
=============================================================

PURPOSE
  Generates a WebGL-based 3D UMAP viewer for scRNAseq datasets. The viewer shows
  continuous UMAP warping by sweeping n_neighbors, with Procrustes-aligned keyframes
  interpolated via cubic splines, rendered as glowing point-light sources.

INPUT FORMATS
  1. h5ad (AnnData) — any scanpy-processed .h5ad file with PCA in obsm['X_pca']
     Usage: python build_viewer.py --input dataset.h5ad --output-dir viewer/datasets/name
  2. Export directory — legacy format from R/Seurat export (pca.csv, metadata.csv, etc.)
     Usage: python build_viewer.py --output-dir viewer/datasets/name

OUTPUT FILES  (written to --output-dir)
  frames.bin        - Float32: nFrames x nCells x 3 positions (ping-pong sweep)
  colors.bin        - Float32: nCells x 3 RGB (cell_type palette)
  metadata.json     - Cell metadata, palette, nn_labels, gene_names, metadata_cols
  expression.bin    - Uint8: all genes concatenated (quantized 0-255)
  gene_index.json   - {gene: [byte_offset, byte_length]} for Range requests
  gene_ranks.json   - {gene: rank} by total expression (built separately)

PIPELINE
  1. Load data from h5ad or export directory
  2. Quantize expression to uint8, write expression.bin + gene_index.json
  3. Compute 3D UMAP keyframes sweeping n_neighbors (5..150)
  4. Procrustes-align all keyframes to the middle one
  5. Cubic spline interpolation -> smooth frames, then ping-pong (forward+reverse)
  6. Write frames.bin, colors.bin, metadata.json

Serve with: python -m http.server 8000 --directory viewer
Open: http://localhost:8000
"""

import numpy as np
import pandas as pd
import umap
from scipy.interpolate import CubicSpline
from scipy.sparse import csc_matrix, issparse
import json
import os
import struct
import argparse

parser = argparse.ArgumentParser(description="Build scAnimator dataset")
parser.add_argument("--input", default=None,
                    help="Input .h5ad file or export directory (default: legacy export/)")
parser.add_argument("--output-dir", default="C:/Users/Nautilus/Desktop/scRNAseq/viewer",
                    help="Output directory for dataset files")
parser.add_argument("--name", default=None,
                    help="Dataset display name (stored in metadata.json)")
parser.add_argument("--cell-type-col", default=None,
                    help="Column in .obs for cell type labels (auto-detected if omitted)")
parser.add_argument("--metadata-cols", default=None,
                    help="Comma-separated .obs columns to include (default: all categorical)")
parser.add_argument("--filter-region", default=None, choices=["dorsal", "ventral"],
                    help="Filter cells to a specific region (legacy export only)")
args = parser.parse_args()

OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)


def procrustes_align(reference, target):
    mu_ref = reference.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    ref_c = reference - mu_ref
    tgt_c = target - mu_tgt
    scale_ref = np.sqrt((ref_c ** 2).sum())
    scale_tgt = np.sqrt((tgt_c ** 2).sum())
    tgt_c *= scale_ref / scale_tgt
    H = tgt_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)
    R = Vt.T @ sign_matrix @ U.T
    return tgt_c @ R.T + mu_ref


def detect_cell_type_col(obs_columns):
    """Auto-detect the cell type column from common naming conventions."""
    candidates = ["cell_type", "celltype", "cell_type_ontology_term_id",
                   "CellType", "Cell_Type", "leiden", "louvain",
                   "cluster", "clusters", "annotation", "cell_annotation"]
    for c in candidates:
        if c in obs_columns:
            return c
    return None


def load_from_h5ad(path, args):
    """Load data from an h5ad (AnnData) file."""
    import scanpy as sc

    print(f"Loading h5ad: {path}")
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # PCA
    if "X_pca" not in adata.obsm:
        raise ValueError("No PCA found in adata.obsm['X_pca']. "
                         "Run sc.tl.pca(adata) before saving the h5ad.")
    X_pca = adata.obsm["X_pca"].astype(np.float32)
    if X_pca.shape[1] > 50:
        X_pca = X_pca[:, :50]
    print(f"  PCA: {X_pca.shape}")

    # Expression matrix (genes x cells, CSC)
    # Prefer .raw if it exists and has more genes (full gene set before HVG filtering)
    if adata.raw is not None and adata.raw.shape[1] > adata.shape[1]:
        print(f"  Using .raw expression ({adata.raw.shape[1]} genes)")
        expr_mat = adata.raw.X  # cells x genes
        gene_names = list(adata.raw.var_names)
    else:
        expr_mat = adata.X  # cells x genes
        gene_names = list(adata.var_names)

    # Convert to CSC (genes x cells) to match existing pipeline
    if issparse(expr_mat):
        expr_sparse = csc_matrix(expr_mat.T)
    else:
        expr_sparse = csc_matrix(expr_mat.T)
    print(f"  Expression: {expr_sparse.shape[0]} genes x {expr_sparse.shape[1]} cells")

    # Cell type column
    ct_col = args.cell_type_col or detect_cell_type_col(adata.obs.columns)
    if ct_col is None:
        raise ValueError("Could not auto-detect cell type column. "
                         f"Available columns: {list(adata.obs.columns)}. "
                         "Use --cell-type-col to specify.")
    if ct_col not in adata.obs.columns:
        raise ValueError(f"Column '{ct_col}' not found in .obs. "
                         f"Available: {list(adata.obs.columns)}")
    print(f"  Cell type column: {ct_col}")

    # Metadata — select columns
    if args.metadata_cols:
        meta_col_names = [c.strip() for c in args.metadata_cols.split(",")]
        missing = [c for c in meta_col_names if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"Columns not found in .obs: {missing}")
    else:
        # Auto-detect: all categorical/string/object columns
        meta_col_names = []
        for col in adata.obs.columns:
            dtype = adata.obs[col].dtype
            if hasattr(dtype, "name") and dtype.name == "category":
                meta_col_names.append(col)
            elif dtype == object:
                meta_col_names.append(col)
        # Ensure cell type column is included
        if ct_col not in meta_col_names:
            meta_col_names.insert(0, ct_col)

    meta = adata.obs[meta_col_names].copy()
    cell_types = adata.obs[ct_col].values.astype(str)
    print(f"  Metadata columns: {meta_col_names}")

    return X_pca, meta, gene_names, expr_sparse, cell_types, ct_col, meta_col_names


def load_from_export(args):
    """Load data from legacy Seurat export directory."""
    EXPORT_DIR = "C:/Users/Nautilus/Desktop/scRNAseq/export"

    print("Loading exported data...")
    pca_df = pd.read_csv(os.path.join(EXPORT_DIR, "pca.csv"), index_col=0)
    X_pca = pca_df.values.astype(np.float32)
    print(f"  PCA: {X_pca.shape}")

    meta = pd.read_csv(os.path.join(EXPORT_DIR, "metadata.csv"), index_col=0)

    if args.filter_region:
        mask = meta["region"].values == args.filter_region
        meta = meta[mask].copy()
        pca_df = pca_df[mask].copy()
        X_pca = pca_df.values.astype(np.float32)
        print(f"  Filtered to region={args.filter_region}: {X_pca.shape[0]} cells")

    cell_types = meta["cell_type"].values.astype(str)
    ct_col = "cell_type"
    meta_col_names = ["cell_type", "seurat_clusters", "orig.ident", "region", "subgroup"]
    meta_col_names = [c for c in meta_col_names if c in meta.columns]

    print("Loading sparse expression...")
    gene_names = open(os.path.join(EXPORT_DIR, "gene_names.txt")).read().strip().split("\n")
    dims = np.fromfile(os.path.join(EXPORT_DIR, "expr_dim.bin"), dtype=np.int32)
    n_genes, n_cells_expr = int(dims[0]), int(dims[1])
    expr_i = np.fromfile(os.path.join(EXPORT_DIR, "expr_i.bin"), dtype=np.int32)
    expr_p = np.fromfile(os.path.join(EXPORT_DIR, "expr_p.bin"), dtype=np.int32)
    expr_x = np.fromfile(os.path.join(EXPORT_DIR, "expr_x.bin"), dtype=np.float32)
    expr_sparse = csc_matrix((expr_x, expr_i, expr_p), shape=(n_genes, n_cells_expr))
    print(f"  Expression: {n_genes} genes x {n_cells_expr} cells")

    if args.filter_region:
        cell_indices = np.where(mask)[0]
        expr_sparse = expr_sparse[:, cell_indices]
        print(f"  Expression filtered: {expr_sparse.shape[0]} genes x {expr_sparse.shape[1]} cells")

    return X_pca, meta, gene_names, expr_sparse, cell_types, ct_col, meta_col_names


# ── Load data ──────────────────────────────────────────────
if args.input and args.input.endswith(".h5ad"):
    X_pca, meta, gene_names, expr_sparse, cell_types, ct_col, meta_col_names = \
        load_from_h5ad(args.input, args)
else:
    X_pca, meta, gene_names, expr_sparse, cell_types, ct_col, meta_col_names = \
        load_from_export(args)

n_cells = X_pca.shape[0]
n_genes = expr_sparse.shape[0]
n_cells_expr = expr_sparse.shape[1]
unique_types = sorted(set(cell_types))
print(f"  Cells: {n_cells}, Genes: {n_genes}, Cell types: {len(unique_types)}")

# ── Quantize and write expression.bin + gene_index.json ─────
print("Writing expression data (batch mode)...")
gene_index = {}
offset = 0
batch_size = 500
with open(os.path.join(OUT_DIR, "expression.bin"), "wb") as f:
    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        # Extract batch as dense array (much faster than row-by-row)
        batch = expr_sparse[start:end, :].toarray()  # (batch_size, n_cells)
        for local_i, gi in enumerate(range(start, end)):
            col = batch[local_i]
            mn, mx = col.min(), col.max()
            if mx > mn:
                quantized = ((col - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                quantized = np.zeros(n_cells_expr, dtype=np.uint8)
            f.write(quantized.tobytes())
            gene_index[gene_names[gi]] = [offset, n_cells_expr]
            offset += n_cells_expr
        print(f"  {end}/{n_genes} genes...", flush=True)

print(f"  expression.bin: {offset / 1024 / 1024:.1f} MB")

with open(os.path.join(OUT_DIR, "gene_index.json"), "w") as f:
    json.dump(gene_index, f, separators=(",", ":"))
print(f"  gene_index.json: {len(gene_index)} genes")

# ── Build gene_ranks.json (rank by total expression) ──────
print("Computing gene ranks...")
gene_totals = np.array(expr_sparse.sum(axis=1)).flatten()
ranked_indices = np.argsort(-gene_totals)
gene_ranks = {}
for rank, idx in enumerate(ranked_indices, 1):
    gene_ranks[gene_names[idx]] = rank
with open(os.path.join(OUT_DIR, "gene_ranks.json"), "w") as f:
    json.dump(gene_ranks, f, separators=(",", ":"))
print(f"  gene_ranks.json: {len(gene_ranks)} genes ranked")

# ── Compute UMAP sweep (n_neighbors only) ───────────────────
nn_values = np.array([5, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150])
n_interp_frames = 180

print(f"Computing {len(nn_values)} UMAP keyframes...")
embeddings = {}
for nn in nn_values:
    print(f"  n_neighbors={nn}")
    reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.3, random_state=42)
    embeddings[nn] = reducer.fit_transform(X_pca)

ref_nn = 40
print(f"Aligning to n_neighbors={ref_nn}...")
reference = embeddings[ref_nn]
aligned = {}
for nn in nn_values:
    if nn == ref_nn:
        aligned[nn] = reference.copy()
    else:
        aligned[nn] = procrustes_align(reference, embeddings[nn])

print(f"Interpolating {n_interp_frames} frames...")
nn_fine = np.linspace(nn_values[0], nn_values[-1], n_interp_frames)
keyframe_stack = np.array([aligned[nn] for nn in nn_values])

interp_stack = np.zeros((n_interp_frames, n_cells, 3), dtype=np.float32)
for dim in range(3):
    for cell_i in range(n_cells):
        cs = CubicSpline(nn_values.astype(float), keyframe_stack[:, cell_i, dim])
        interp_stack[:, cell_i, dim] = cs(nn_fine)

# Normalize
center = interp_stack.reshape(-1, 3).mean(axis=0)
interp_stack -= center
scale = np.abs(interp_stack.reshape(-1, 3)).max()
interp_stack /= scale
interp_stack *= 5

# Ping-pong
interp_pp = np.concatenate([interp_stack, interp_stack[-2:0:-1]], axis=0).astype(np.float32)
nn_fine_pp = np.concatenate([nn_fine, nn_fine[-2:0:-1]])
n_frames = interp_pp.shape[0]
print(f"  {n_frames} total frames (ping-pong)")

# ── Write frames.bin ────────────────────────────────────────
frames_path = os.path.join(OUT_DIR, "frames.bin")
interp_pp.tofile(frames_path)
print(f"  frames.bin: {os.path.getsize(frames_path) / 1024 / 1024:.1f} MB")

# ── Write colors.bin ────────────────────────────────────────
palette_hex = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a",
]
palette = {}
for i, ct in enumerate(unique_types):
    h = palette_hex[i % len(palette_hex)]
    palette[ct] = [int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255]

colors = np.array([palette.get(ct, [0.5, 0.5, 0.5]) for ct in cell_types], dtype=np.float32)
colors_path = os.path.join(OUT_DIR, "colors.bin")
colors.tofile(colors_path)
print(f"  colors.bin: {os.path.getsize(colors_path) / 1024 / 1024:.1f} MB")

# ── Build metadata_cols ────────────────────────────────────
# Known pretty labels for common columns; others get title-cased
meta_col_labels = {
    "cell_type": "Cell Type",
    "celltype": "Cell Type",
    "seurat_clusters": "Seurat Cluster",
    "orig.ident": "Sample",
    "region": "Region",
    "subgroup": "Subgroup",
    "leiden": "Leiden Cluster",
    "louvain": "Louvain Cluster",
    "batch": "Batch",
    "sample": "Sample",
    "tissue": "Tissue",
    "donor": "Donor",
}
metadata_cols = {}
for col in meta_col_names:
    if col in meta.columns:
        vals = meta[col].values.astype(str).tolist()
        uniq = sorted(set(vals))
        label = meta_col_labels.get(col, col.replace("_", " ").title())
        metadata_cols[col] = {
            "label": label,
            "values": vals,
            "unique": uniq,
        }

# ── Write metadata.json ────────────────────────────────────
metadata = {
    "n_cells": n_cells,
    "n_frames": n_frames,
    "nn_labels": [f"{v:.0f}" for v in nn_fine_pp],
    "cell_types": cell_types.tolist(),
    "unique_types": unique_types,
    "palette": palette,
    "gene_names": gene_names,
    "metadata_cols": metadata_cols,
}
if args.name:
    metadata["name"] = args.name
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, separators=(",", ":"))
print(f"  metadata.json written")

# ── Summary ────────────────────────────────────────────────

print(f"\n=== Output ({OUT_DIR}) ===")
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp)
        print(f"  {fn}: {sz / 1024 / 1024:.1f} MB" if sz > 1024*1024 else f"  {fn}: {sz / 1024:.1f} KB")

print(f"\nDone.")
