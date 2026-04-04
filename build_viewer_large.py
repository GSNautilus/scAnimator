"""
build_viewer_large.py -- Memory-efficient builder for large scRNAseq datasets.
===============================================================================

Like build_viewer.py but optimized for datasets with 1M+ cells:
  - Backed-mode h5ad reading (expression stays on disk)
  - Single static 3D UMAP (no animation sweep)
  - h5py-based chunked expression quantization (row-major reading)
  - Supports custom embedding keys (e.g. X_scpoli instead of X_pca)

Usage:
  python build_viewer_large.py \
    --input dataset.h5ad \
    --output-dir viewer/datasets/brain_organoid \
    --name "Human Brain Organoid Atlas" \
    --embedding-key X_scpoli \
    --cell-type-col annot_level_3_rev2 \
    --metadata-cols annot_level_1,annot_level_2,annot_level_3_rev2,annot_region_rev2,cell_type \
    --hvg-only
"""

import numpy as np
import json
import os
import argparse
import time
import gc

parser = argparse.ArgumentParser(description="Build scAnimator dataset (large/static)")
parser.add_argument("--input", required=True, help="Input .h5ad file")
parser.add_argument("--output-dir", required=True, help="Output directory")
parser.add_argument("--name", default=None, help="Dataset display name")
parser.add_argument("--embedding-key", default="X_pca",
                    help="obsm key for UMAP input (default: X_pca)")
parser.add_argument("--cell-type-col", default=None,
                    help="Column in .obs for cell type labels")
parser.add_argument("--metadata-cols", default=None,
                    help="Comma-separated .obs columns to include")
parser.add_argument("--hvg-only", action="store_true",
                    help="Only include highly variable genes in expression.bin")
parser.add_argument("--umap-neighbors", type=int, default=15,
                    help="n_neighbors for single static UMAP (default: 15)")
parser.add_argument("--sweep", default=None,
                    help="Enable UMAP sweep mode: 'start-end-nkeys' e.g. '5-50-20' "
                         "for 20 evenly spaced keyframes from nn=5 to nn=50")
parser.add_argument("--n-interp-frames", type=int, default=180,
                    help="Number of interpolation frames for sweep (default: 180)")
parser.add_argument("--skip-umap", action="store_true",
                    help="Skip UMAP computation (reuse existing frames.bin)")
parser.add_argument("--skip-expression", action="store_true",
                    help="Skip expression quantization (reuse existing expression.bin)")
args = parser.parse_args()

OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()


def elapsed():
    return f"[{time.time() - t0:.0f}s]"


# -- Step 1: Load h5ad in backed mode ----------------------
print(f"{elapsed()} Loading h5ad (backed mode): {args.input}")
import scanpy as sc

adata = sc.read_h5ad(args.input, backed='r')
n_cells = adata.shape[0]
n_genes_total = adata.shape[1]
print(f"  Shape: {n_cells:,} cells x {n_genes_total:,} genes")
print(f"  obsm keys: {list(adata.obsm.keys())}")

# -- Embedding for UMAP -----------------------------------
emb_key = args.embedding_key
if emb_key not in adata.obsm:
    available = list(adata.obsm.keys())
    raise ValueError(f"Embedding '{emb_key}' not in obsm. Available: {available}")

X_emb = np.array(adata.obsm[emb_key], dtype=np.float32)
print(f"  Embedding: {emb_key} {X_emb.shape}")

# -- Cell type column --------------------------------------
ct_candidates = ["cell_type", "celltype", "CellType", "annot_level_3_rev2",
                 "annot_level_1", "leiden", "louvain", "cluster", "annotation"]
ct_col = args.cell_type_col
if ct_col is None:
    for c in ct_candidates:
        if c in adata.obs.columns:
            ct_col = c
            break
if ct_col is None or ct_col not in adata.obs.columns:
    raise ValueError(f"Cell type column not found. Available: {list(adata.obs.columns)}")

cell_types = adata.obs[ct_col].values.astype(str)
unique_types = sorted(set(cell_types))
print(f"  Cell type column: {ct_col} ({len(unique_types)} unique)")

# -- Metadata columns -------------------------------------
if args.metadata_cols:
    meta_col_names = [c.strip() for c in args.metadata_cols.split(",")]
    missing = [c for c in meta_col_names if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
else:
    meta_col_names = []
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype
        if hasattr(dtype, "name") and dtype.name == "category":
            n_unique = adata.obs[col].nunique()
            if n_unique <= 500:
                meta_col_names.append(col)
    if ct_col not in meta_col_names:
        meta_col_names.insert(0, ct_col)

print(f"  Metadata columns: {meta_col_names}")

# -- Determine genes to include ----------------------------
if args.hvg_only and 'highly_variable' in adata.var.columns:
    hvg_mask = adata.var['highly_variable'].values
    gene_indices = np.where(hvg_mask)[0]
    if 'feature_name' in adata.var.columns:
        all_gene_names = adata.var['feature_name'].values.astype(str)
    else:
        all_gene_names = np.array(adata.var_names, dtype=str)
    gene_names = [all_gene_names[i] for i in gene_indices]
    print(f"  HVG-only mode: {len(gene_names)} / {n_genes_total} genes")
else:
    gene_indices = np.arange(n_genes_total)
    if 'feature_name' in adata.var.columns:
        gene_names = list(adata.var['feature_name'].values.astype(str))
    else:
        gene_names = list(adata.var_names)
    print(f"  All genes: {len(gene_names)}")

n_genes = len(gene_names)

# -- Collect metadata before closing adata -----------------
meta_col_labels = {
    "cell_type": "Cell Type",
    "annot_level_1": "Annotation Level 1",
    "annot_level_2": "Annotation Level 2",
    "annot_level_3_rev2": "Annotation Level 3",
    "annot_region_rev2": "Brain Region",
    "organoid_age_days": "Organoid Age (days)",
    "publication": "Publication",
    "cell_line": "Cell Line",
    "batch": "Batch",
    "tissue": "Tissue",
    "disease": "Disease",
    "sex": "Sex",
    "assay": "Assay",
}
metadata_cols = {}
for col in meta_col_names:
    if col in adata.obs.columns:
        vals = adata.obs[col].values.astype(str).tolist()
        uniq = sorted(set(vals))
        label = meta_col_labels.get(col, col.replace("_", " ").title())
        metadata_cols[col] = {
            "label": label,
            "values": vals,
            "unique": uniq,
        }

# Close backed adata to free memory and file handle
del adata
gc.collect()
print(f"{elapsed()} Closed backed adata, metadata collected.")

# -- Step 2: Compute 3D UMAP ------------------------------
def procrustes_align(reference, target):
    """Rigid alignment (rotation + reflection + scale) of target to reference."""
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


if args.skip_umap:
    # Determine n_frames from existing frames.bin
    frames_path = os.path.join(OUT_DIR, "frames.bin")
    frames_size = os.path.getsize(frames_path)
    n_frames = frames_size // (n_cells * 3 * 4)
    print(f"\n{elapsed()} Skipping UMAP (--skip-umap), reusing frames.bin ({n_frames} frames)")

elif args.sweep:
    # Parse sweep: "start-end-nkeys" e.g. "5-50-20"
    parts = args.sweep.split("-")
    nn_start, nn_end, n_keys = int(parts[0]), int(parts[1]), int(parts[2])
    nn_values = np.unique(np.linspace(nn_start, nn_end, n_keys).astype(int))
    n_interp = args.n_interp_frames

    print(f"\n{elapsed()} UMAP sweep: {len(nn_values)} keyframes from nn={nn_values[0]} to nn={nn_values[-1]}")
    print(f"  Keyframes: {nn_values.tolist()}")
    print(f"  ~{len(nn_values) * 20} minutes estimated. Each keyframe ~20 min.")

    import umap

    # Checkpoint directory for crash recovery
    ckpt_dir = os.path.join(OUT_DIR, "_keyframe_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    embeddings = {}
    for ki, nn in enumerate(nn_values):
        ckpt_path = os.path.join(ckpt_dir, f"keyframe_nn{nn}.npy")
        if os.path.exists(ckpt_path):
            print(f"\n{elapsed()} Keyframe {ki+1}/{len(nn_values)}: n_neighbors={nn} (loaded from checkpoint)")
            embeddings[nn] = np.load(ckpt_path)
            continue

        print(f"\n{elapsed()} Keyframe {ki+1}/{len(nn_values)}: n_neighbors={nn}")
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=int(nn),
            min_dist=0.3,
            random_state=42,
            low_memory=True,
            init='random',
            verbose=True,
        )
        embeddings[nn] = reducer.fit_transform(X_emb)
        np.save(ckpt_path, embeddings[nn])
        print(f"  Checkpoint saved: {ckpt_path}")
        del reducer
        gc.collect()

    del X_emb
    gc.collect()

    # Procrustes align to middle keyframe
    ref_nn = nn_values[len(nn_values) // 2]
    print(f"\n{elapsed()} Aligning to n_neighbors={ref_nn}...")
    reference = embeddings[ref_nn]
    aligned = {}
    for nn in nn_values:
        if nn == ref_nn:
            aligned[nn] = reference.copy()
        else:
            aligned[nn] = procrustes_align(reference, embeddings[nn])
    del embeddings
    gc.collect()

    # Cubic spline interpolation
    from scipy.interpolate import CubicSpline
    print(f"{elapsed()} Interpolating {n_interp} frames...")
    nn_fine = np.linspace(nn_values[0], nn_values[-1], n_interp)
    keyframe_stack = np.array([aligned[nn] for nn in nn_values])
    del aligned
    gc.collect()

    interp_stack = np.zeros((n_interp, n_cells, 3), dtype=np.float32)
    for dim in range(3):
        for cell_i in range(n_cells):
            cs = CubicSpline(nn_values.astype(float), keyframe_stack[:, cell_i, dim])
            interp_stack[:, cell_i, dim] = cs(nn_fine)
        print(f"  dim {dim}/3 done... {elapsed()}")

    del keyframe_stack
    gc.collect()

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
    nn_labels = [f"{v:.0f}" for v in nn_fine_pp]
    print(f"  {n_frames} total frames (ping-pong)")

    del interp_stack
    gc.collect()

    frames_path = os.path.join(OUT_DIR, "frames.bin")
    interp_pp.tofile(frames_path)
    print(f"{elapsed()} frames.bin: {os.path.getsize(frames_path) / 1024 / 1024:.1f} MB")

    del interp_pp
    gc.collect()

else:
    # Single static UMAP
    print(f"\n{elapsed()} Computing 3D UMAP ({n_cells:,} cells, n_neighbors={args.umap_neighbors})...")
    print("  This may take 15-45 minutes for 1M+ cells. Watch RAM in Task Manager.")

    import umap
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=0.3,
        random_state=42,
        low_memory=True,
        init='random',
        verbose=True,
    )
    coords_3d = reducer.fit_transform(X_emb)
    del X_emb, reducer
    gc.collect()

    center = coords_3d.mean(axis=0)
    coords_3d -= center
    scale = np.abs(coords_3d).max()
    coords_3d /= scale
    coords_3d *= 5
    coords_3d = coords_3d.astype(np.float32)

    frames = coords_3d.reshape(1, n_cells, 3)
    n_frames = 1

    frames_path = os.path.join(OUT_DIR, "frames.bin")
    frames.tofile(frames_path)
    print(f"{elapsed()} frames.bin: {os.path.getsize(frames_path) / 1024 / 1024:.1f} MB")

    del frames, coords_3d
    gc.collect()

# Build nn_labels for metadata
if not args.skip_umap and not args.sweep:
    nn_labels = ["static"]
elif args.skip_umap:
    nn_labels = ["static"]  # will be overridden if sweep frames exist

# -- Step 3: Write colors.bin -----------------------------
print(f"{elapsed()} Writing colors.bin...")
palette_hex = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31",
]

palette = {}
for i, ct in enumerate(unique_types):
    h = palette_hex[i % len(palette_hex)]
    palette[ct] = [int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255]

colors = np.array([palette.get(ct, [0.5, 0.5, 0.5]) for ct in cell_types], dtype=np.float32)
colors_path = os.path.join(OUT_DIR, "colors.bin")
colors.tofile(colors_path)
print(f"  colors.bin: {os.path.getsize(colors_path) / 1024 / 1024:.1f} MB")
del colors
gc.collect()

# -- Step 4: Quantize expression via h5py chunked reading --
if args.skip_expression:
    print(f"\n{elapsed()} Skipping expression (--skip-expression)")
else:
    print(f"\n{elapsed()} Expression quantization ({n_genes:,} genes via h5py chunked reading)...")

    from scipy.sparse import csr_matrix
    import h5py

    # Allocate gene-major buffer: (n_genes, n_cells) float32
    buf_gb = n_genes * n_cells * 4 / 1024**3
    print(f"  Allocating expression buffer: {n_genes} x {n_cells:,} float32 = {buf_gb:.1f} GB")
    expr_buf = np.zeros((n_genes, n_cells), dtype=np.float32)

    cell_chunk_size = 50000

    with h5py.File(args.input, 'r') as f:
        X_grp = f['X']
        encoding = X_grp.attrs.get('encoding-type', b'csr_matrix')
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        print(f"  Sparse encoding: {encoding}")

        data_ds = X_grp['data']
        indices_ds = X_grp['indices']
        indptr_ds = X_grp['indptr']

        # Read indptr fully (small: n_cells+1 entries, ~14 MB)
        print(f"  Reading indptr ({indptr_ds.shape[0]:,} entries)...")
        indptr_full = indptr_ds[:]

        n_chunks = (n_cells + cell_chunk_size - 1) // cell_chunk_size
        print(f"  Processing {n_chunks} cell chunks of {cell_chunk_size:,}...")

        for ci, chunk_start in enumerate(range(0, n_cells, cell_chunk_size)):
            chunk_end = min(chunk_start + cell_chunk_size, n_cells)
            n_chunk_cells = chunk_end - chunk_start

            # Range of non-zero entries for these rows
            ptr_start = int(indptr_full[chunk_start])
            ptr_end = int(indptr_full[chunk_end])

            # Read only the relevant slice of data and indices
            chunk_data = data_ds[ptr_start:ptr_end]
            chunk_indices = indices_ds[ptr_start:ptr_end]

            # Build local CSR
            local_indptr = indptr_full[chunk_start:chunk_end + 1] - ptr_start
            chunk_csr = csr_matrix(
                (chunk_data, chunk_indices, local_indptr),
                shape=(n_chunk_cells, n_genes_total)
            )

            # Extract HVG columns and store transposed
            chunk_hvg = chunk_csr[:, gene_indices].toarray().astype(np.float32)
            expr_buf[:, chunk_start:chunk_end] = chunk_hvg.T

            del chunk_data, chunk_indices, chunk_csr, chunk_hvg
            gc.collect()

            if (ci + 1) % 5 == 0 or chunk_end == n_cells:
                print(f"    {chunk_end:,}/{n_cells:,} cells... {elapsed()}", flush=True)

    # Quantize and write expression.bin
    print(f"{elapsed()} Writing expression.bin...")
    gene_index = {}
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    offset = 0

    with open(os.path.join(OUT_DIR, "expression.bin"), "wb") as f:
        for gi in range(n_genes):
            col = expr_buf[gi]
            mn, mx = col.min(), col.max()
            gene_totals[gi] = col.sum()
            if mx > mn:
                quantized = ((col - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                quantized = np.zeros(n_cells, dtype=np.uint8)
            f.write(quantized.tobytes())
            gene_index[gene_names[gi]] = [offset, n_cells]
            offset += n_cells

            if (gi + 1) % 500 == 0 or gi == n_genes - 1:
                print(f"  {gi + 1}/{n_genes} genes written... {elapsed()}", flush=True)

    del expr_buf
    gc.collect()

    expr_size = os.path.getsize(os.path.join(OUT_DIR, "expression.bin"))
    print(f"  expression.bin: {expr_size / 1024 / 1024:.1f} MB")

    with open(os.path.join(OUT_DIR, "gene_index.json"), "w") as f:
        json.dump(gene_index, f, separators=(",", ":"))
    print(f"  gene_index.json: {len(gene_index)} genes")

    # Gene ranks
    ranked_indices = np.argsort(-gene_totals)
    gene_ranks = {}
    for rank, idx in enumerate(ranked_indices, 1):
        gene_ranks[gene_names[idx]] = rank
    with open(os.path.join(OUT_DIR, "gene_ranks.json"), "w") as f:
        json.dump(gene_ranks, f, separators=(",", ":"))
    print(f"  gene_ranks.json: {len(gene_ranks)} genes ranked")

# -- Step 5: Metadata JSON --------------------------------
print(f"{elapsed()} Writing metadata.json...")

metadata = {
    "n_cells": n_cells,
    "n_frames": n_frames,
    "nn_labels": nn_labels,
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
meta_size = os.path.getsize(os.path.join(OUT_DIR, "metadata.json"))
print(f"  metadata.json: {meta_size / 1024 / 1024:.1f} MB")

# -- Summary -----------------------------------------------
print(f"\n{elapsed()} === Output ({OUT_DIR}) ===")
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp)
        if sz > 1024 * 1024:
            print(f"  {fn}: {sz / 1024 / 1024:.1f} MB")
        else:
            print(f"  {fn}: {sz / 1024:.1f} KB")

print(f"\n{elapsed()} Done. {n_cells:,} cells, {n_genes:,} genes, {n_frames} frame(s).")
