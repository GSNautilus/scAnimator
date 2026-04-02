"""
Neural scRNAseq: Build UMAP warp viewer with external data files.
=================================================================

PURPOSE
  Generates a WebGL-based 3D UMAP viewer for a ~45K-cell neural scRNAseq dataset
  (Seurat object from 202410.RData). The viewer shows continuous UMAP warping by
  sweeping n_neighbors, with Procrustes-aligned keyframes interpolated via cubic
  splines, rendered as glowing point-light sources on a black background.

SOURCE DATA  (from R export at scRNAseq/export/)
  pca.csv         - 50-dim PCA embeddings (45386 cells)
  metadata.csv    - Cell metadata: cell_type (scType), seurat_clusters, orig.ident,
                    region (dorsal/ventral), subgroup (B/C)
  gene_names.txt  - 28134 gene symbols
  expr_*.bin      - Sparse CSC expression matrix (genes x cells)

OUTPUT FILES  (written to scRNAseq/viewer/)
  index.html        - Three.js viewer (see header in that file for architecture)
  frames.bin        - Float32: nFrames x nCells x 3 positions (ping-pong sweep)
  colors.bin        - Float32: nCells x 3 RGB (cell_type palette)
  metadata.json     - Cell metadata, palette, nn_labels, gene_names,
                      metadata_cols (cell_type, seurat_clusters, orig.ident,
                      region, subgroup — each with label, values[], unique[])
  expression.bin    - Uint8: all 28134 genes concatenated (quantized 0-255)
  gene_index.json   - {gene: [byte_offset, byte_length]} for Range requests
  gene_ranks.json   - {gene: rank} by total expression (1=highest). Built by
                      a separate script, not this one.

PIPELINE
  1. Load PCA + metadata from export/
  2. Load sparse expression, quantize to uint8, write expression.bin + gene_index.json
  3. Compute 3D UMAP keyframes sweeping n_neighbors (5..150, ~20 keyframes)
  4. Procrustes-align all keyframes to the middle one (rotation, scale, translation)
  5. Cubic spline interpolation -> 358 smooth frames, then ping-pong (forward+reverse)
  6. Write frames.bin, colors.bin, metadata.json
  7. Write index.html (embedded as a Python string at the bottom — BUT the live
     viewer/index.html is edited directly and may be newer than this embedded copy)

IMPORTANT NOTES
  - The embedded HTML in this script may be STALE. The live viewer is edited in-place
    at viewer/index.html. Always check that file for the current viewer code.
  - OrbitControls: enablePan=false, target locked at origin
  - Only n_neighbors sweep (no min_dist sweep for this dataset)
  - Expression writing uses batch mode (500 genes at a time) for performance
  - Conda env: codex (or direct: /c/Users/Nautilus/miniconda3/envs/codex/python.exe)

Serve with: python -m http.server 8000 --directory viewer
Open: http://localhost:8000
"""

import numpy as np
import pandas as pd
import umap
from scipy.interpolate import CubicSpline
from scipy.sparse import csc_matrix
import json
import os
import struct
import argparse

parser = argparse.ArgumentParser(description="Build scAnimator dataset")
parser.add_argument("--output-dir", default="C:/Users/Nautilus/Desktop/scRNAseq/viewer",
                    help="Output directory for dataset files")
parser.add_argument("--name", default=None,
                    help="Dataset display name (stored in metadata.json)")
parser.add_argument("--filter-region", default=None, choices=["dorsal", "ventral"],
                    help="Filter cells to a specific region")
args = parser.parse_args()

EXPORT_DIR = "C:/Users/Nautilus/Desktop/scRNAseq/export"
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


# ── Load exported data ──────────────────────────────────────
print("Loading exported data...")
pca_df = pd.read_csv(os.path.join(EXPORT_DIR, "pca.csv"), index_col=0)
X_pca = pca_df.values.astype(np.float32)
n_cells = X_pca.shape[0]
print(f"  PCA: {X_pca.shape}")

meta = pd.read_csv(os.path.join(EXPORT_DIR, "metadata.csv"), index_col=0)

# ── Optional region filter ─────────────────────────────────
if args.filter_region:
    mask = meta["region"].values == args.filter_region
    meta = meta[mask].copy()
    pca_df = pca_df[mask].copy()
    X_pca = pca_df.values.astype(np.float32)
    n_cells = X_pca.shape[0]
    print(f"  Filtered to region={args.filter_region}: {n_cells} cells")

cell_types = meta["cell_type"].values.astype(str)
seurat_clusters = meta["seurat_clusters"].values.astype(str)
unique_types = sorted(set(cell_types))
print(f"  Cells: {n_cells}, Cell types: {len(unique_types)}")

# ── Load sparse expression ──────────────────────────────────
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
    n_cells_expr = expr_sparse.shape[1]
    print(f"  Expression filtered: {n_genes} genes x {n_cells_expr} cells")

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
meta_col_names = ["cell_type", "seurat_clusters", "orig.ident", "region", "subgroup"]
meta_col_labels = {
    "cell_type": "Cell Type",
    "seurat_clusters": "Seurat Cluster",
    "orig.ident": "Sample",
    "region": "Region",
    "subgroup": "Subgroup",
}
metadata_cols = {}
for col in meta_col_names:
    if col in meta.columns:
        vals = meta[col].values.astype(str).tolist()
        uniq = sorted(set(vals))
        metadata_cols[col] = {
            "label": meta_col_labels.get(col, col),
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
