"""
scAnimator: Build UMAP warp viewer dataset.
============================================

Generates a WebGL-based 3D UMAP viewer for scRNAseq datasets. Handles both
small (< 100k cells) and large (1M+) datasets efficiently via adaptive
memory strategies.

INPUT FORMATS
  1. h5ad (AnnData) -- any .h5ad file. If no suitable embedding (>= 3D) is
     found in obsm, PCA (50 components) is computed automatically. Use
     --no-pca to skip this and run UMAP directly on expression (slower).
  2. Export directory -- legacy format from R/Seurat export (pca.csv, etc.)

UMAP MODES
  - Static:  single 3D UMAP (default, fast)
  - Sweep:   animated n_neighbors warp with Procrustes + cubic spline
             interpolation and ping-pong playback

OUTPUT FILES  (written to --output-dir)
  frames.bin        Float32: nFrames x nCells x 3 positions
  colors.bin        Float32: nCells x 3 RGB (cell_type palette)
  metadata.json     Cell metadata, palette, nn_labels, gene_names, metadata_cols
  expression.bin    Uint8: genes concatenated (quantized 0-255)
  gene_index.json   {gene: [byte_offset, byte_length]} for Range requests
  gene_ranks.json   {gene: rank} by total expression

EXAMPLES
  # Small dataset, full sweep (original behavior)
  python scripts/build_viewer.py --input pbmc3k.h5ad --output-dir datasets/pbmc3k \\
      --sweep 5-150-11

  # Large dataset, static UMAP, HVGs only
  python scripts/build_viewer.py --input organoid.h5ad --output-dir datasets/organoid \\
      --embedding-key X_scpoli --cell-type-col annot_level_3_rev2 --hvg-only

  # Large dataset, animated sweep with checkpoints
  python scripts/build_viewer.py --input organoid.h5ad --output-dir datasets/organoid \\
      --embedding-key X_scpoli --sweep 5-50-16 --skip-expression

  # Legacy R/Seurat export
  python scripts/build_viewer.py --export-dir ./export --output-dir datasets/neural

Serve with: python -m http.server 8000
"""

import numpy as np
import json
import os
import argparse
import time
import gc

parser = argparse.ArgumentParser(description="Build scAnimator dataset")
parser.add_argument("--input", default=None,
                    help="Input .h5ad file")
parser.add_argument("--export-dir", default=None,
                    help="Legacy R/Seurat export directory (pca.csv, metadata.csv, etc.)")
parser.add_argument("--output-dir", required=True,
                    help="Output directory for dataset files")
parser.add_argument("--name", default=None,
                    help="Dataset display name (stored in metadata.json)")
parser.add_argument("--embedding-key", default="X_pca",
                    help="obsm key for UMAP input (default: X_pca)")
parser.add_argument("--cell-type-col", default=None,
                    help="Column in .obs for cell type labels (auto-detected if omitted)")
parser.add_argument("--metadata-cols", default=None,
                    help="Comma-separated .obs columns to include (default: all categorical)")
parser.add_argument("--hvg-only", action="store_true",
                    help="Only include highly variable genes in expression.bin")
parser.add_argument("--umap-neighbors", type=int, default=15,
                    help="n_neighbors for single static UMAP (default: 15)")
parser.add_argument("--sweep", default=None,
                    help="UMAP sweep: 'start-end-nkeys' e.g. '5-150-11' or '5-50-16'")
parser.add_argument("--n-interp-frames", type=int, default=180,
                    help="Interpolation frames for sweep (default: 180)")
parser.add_argument("--filter-region", default=None, choices=["dorsal", "ventral"],
                    help="Filter cells to a region (legacy export only)")
parser.add_argument("--no-pca", action="store_true",
                    help="Skip auto-PCA fallback; use expression matrix directly for UMAP input")
parser.add_argument("--skip-umap", action="store_true",
                    help="Skip UMAP computation (reuse existing frames.bin)")
parser.add_argument("--skip-expression", action="store_true",
                    help="Skip expression quantization (reuse existing files)")
args = parser.parse_args()

OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()


def elapsed():
    return f"[{time.time() - t0:.0f}s]"


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


def detect_cell_type_col(obs_columns):
    """Auto-detect the cell type column from common naming conventions."""
    candidates = ["cell_type", "celltype", "cell_type_ontology_term_id",
                   "CellType", "Cell_Type", "annot_level_3_rev2",
                   "annot_level_1", "leiden", "louvain",
                   "cluster", "clusters", "annotation", "cell_annotation"]
    for c in candidates:
        if c in obs_columns:
            return c
    return None


# =========================================================================
#  DATA LOADING
# =========================================================================

# Shared state populated by loaders
X_emb = None          # (n_cells, n_dims) float32 embedding for UMAP
cell_types = None     # (n_cells,) str array
unique_types = None
meta_col_names = []
metadata_cols = {}
gene_names = []
gene_indices = None   # indices into the full gene axis (for HVG filtering)
n_cells = 0
n_genes = 0
n_genes_total = 0
h5ad_path = None      # set if h5ad input, needed for expression reading

# Expression source: either 'sparse' (in-memory CSC) or 'h5ad' (backed via h5py)
expr_source = None
expr_sparse = None    # only set for small datasets / legacy export

# Pretty labels for known metadata columns
META_COL_LABELS = {
    "cell_type": "Cell Type", "celltype": "Cell Type",
    "seurat_clusters": "Seurat Cluster", "orig.ident": "Sample",
    "region": "Region", "subgroup": "Subgroup",
    "leiden": "Leiden Cluster", "louvain": "Louvain Cluster",
    "batch": "Batch", "sample": "Sample", "tissue": "Tissue",
    "donor": "Donor", "disease": "Disease", "sex": "Sex", "assay": "Assay",
    "annot_level_1": "Annotation Level 1",
    "annot_level_2": "Annotation Level 2",
    "annot_level_3_rev2": "Annotation Level 3",
    "annot_region_rev2": "Brain Region",
    "organoid_age_days": "Organoid Age (days)",
    "publication": "Publication", "cell_line": "Cell Line",
    "author_celltype": "Author Cell Type", "sub_celltype": "Sub Cell Type",
    "groupid": "GBM Subtype", "donor_id": "Donor", "sample_id": "Sample",
}

# -- Large-cell threshold: above this, use backed mode + h5py expression --
LARGE_THRESHOLD = 200_000


def load_from_h5ad(path):
    """Load data from an h5ad file. Uses backed mode for large datasets."""
    global X_emb, cell_types, unique_types, meta_col_names, metadata_cols
    global gene_names, gene_indices, n_cells, n_genes, n_genes_total
    global expr_source, expr_sparse, h5ad_path
    import scanpy as sc
    from scipy.sparse import csc_matrix, issparse

    h5ad_path = path

    # Peek at size to decide strategy
    adata_peek = sc.read_h5ad(path, backed='r')
    total_cells = adata_peek.shape[0]
    use_backed = total_cells > LARGE_THRESHOLD

    if use_backed:
        print(f"{elapsed()} Loading h5ad (backed mode, {total_cells:,} cells): {path}")
        adata = adata_peek
    else:
        del adata_peek
        print(f"{elapsed()} Loading h5ad (in-memory, {total_cells:,} cells): {path}")
        adata = sc.read_h5ad(path)

    n_cells = adata.shape[0]
    n_genes_total = adata.shape[1]
    print(f"  Shape: {n_cells:,} cells x {n_genes_total:,} genes")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    # Embedding
    emb_key = args.embedding_key
    need_pca = False

    if emb_key not in adata.obsm:
        available = list(adata.obsm.keys())
        # Check if any obsm key has >= 3 dims
        usable = [(k, adata.obsm[k].shape[1]) for k in available if adata.obsm[k].shape[1] >= 3]
        if usable:
            # Pick the best alternative (prefer PCA-like, then highest-dim)
            pca_like = [k for k, d in usable if 'pca' in k.lower()]
            fallback_key = pca_like[0] if pca_like else usable[0][0]
            print(f"  WARNING: '{emb_key}' not in obsm. Using '{fallback_key}' instead.")
            emb_key = fallback_key
        else:
            print(f"  WARNING: '{emb_key}' not in obsm. Available: {available}")
            if available:
                dims = {k: adata.obsm[k].shape[1] for k in available}
                print(f"  Embedding dimensions: {dims}")
                print(f"  All available embeddings are <= 2D — insufficient for 3D UMAP.")
            need_pca = True

    if not need_pca:
        X_emb = np.array(adata.obsm[emb_key], dtype=np.float32)
        if X_emb.shape[1] < 3:
            print(f"  WARNING: Embedding '{emb_key}' is only {X_emb.shape[1]}D — too few dimensions for 3D UMAP.")
            need_pca = True
            del X_emb
        else:
            if X_emb.shape[1] > 50:
                X_emb = X_emb[:, :50]
            print(f"  Embedding: {emb_key} {X_emb.shape}")

    if need_pca:
        if args.no_pca:
            print(f"  --no-pca specified: will run UMAP directly on expression matrix.")
            print(f"  (This is slower and noisier than PCA. Consider removing --no-pca.)")
            from scipy.sparse import issparse
            if use_backed:
                raise ValueError("--no-pca with backed mode not supported. "
                                 "Expression matrix too large to load into memory for direct UMAP.")
            expr_for_umap = adata.X
            if issparse(expr_for_umap):
                expr_for_umap = expr_for_umap.toarray()
            X_emb = np.array(expr_for_umap, dtype=np.float32)
            del expr_for_umap
            print(f"  Using raw expression: {X_emb.shape}")
        else:
            print(f"  Computing PCA (50 components) as UMAP input...")
            import scanpy as sc_inner
            # Need in-memory adata for PCA
            if use_backed:
                print(f"  Loading expression into memory for PCA (large dataset)...")
                adata_mem = sc_inner.read_h5ad(path)
                sc_inner.pp.normalize_total(adata_mem, target_sum=1e4)
                sc_inner.pp.log1p(adata_mem)
                sc_inner.pp.highly_variable_genes(adata_mem, n_top_genes=3000, flavor='seurat',
                                                  subset=False)
                sc_inner.pp.scale(adata_mem, max_value=10)
                sc_inner.tl.pca(adata_mem, n_comps=50, use_highly_variable=True)
                X_emb = np.array(adata_mem.obsm['X_pca'], dtype=np.float32)
                del adata_mem
                gc.collect()
            else:
                # In-memory: run PCA on current adata
                adata_pca = adata.copy()
                sc_inner.pp.normalize_total(adata_pca, target_sum=1e4)
                sc_inner.pp.log1p(adata_pca)
                sc_inner.pp.highly_variable_genes(adata_pca, n_top_genes=3000, flavor='seurat',
                                                  subset=False)
                sc_inner.pp.scale(adata_pca, max_value=10)
                sc_inner.tl.pca(adata_pca, n_comps=50, use_highly_variable=True)
                X_emb = np.array(adata_pca.obsm['X_pca'], dtype=np.float32)
                del adata_pca
                gc.collect()
            print(f"  PCA computed: {X_emb.shape}")

    # Cell type column
    ct_col = args.cell_type_col or detect_cell_type_col(adata.obs.columns)
    if ct_col is None or ct_col not in adata.obs.columns:
        raise ValueError(f"Cell type column not found. Available: {list(adata.obs.columns)}. "
                         "Use --cell-type-col to specify.")
    cell_types = adata.obs[ct_col].values.astype(str)
    unique_types = sorted(set(cell_types))
    print(f"  Cell type column: {ct_col} ({len(unique_types)} unique)")

    # Metadata columns
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
            elif dtype == object:
                meta_col_names.append(col)
        if ct_col not in meta_col_names:
            meta_col_names.insert(0, ct_col)
    print(f"  Metadata columns: {meta_col_names}")

    # Build metadata_cols dict
    for col in meta_col_names:
        if col in adata.obs.columns:
            vals = adata.obs[col].values.astype(str).tolist()
            uniq = sorted(set(vals))
            label = META_COL_LABELS.get(col, col.replace("_", " ").title())
            metadata_cols[col] = {"label": label, "values": vals, "unique": uniq}

    # Genes — detect HVG column (bool 'highly_variable' or int 'vst.variable')
    hvg_col = None
    if args.hvg_only:
        if 'highly_variable' in adata.var.columns:
            hvg_col = 'highly_variable'
        elif 'vst.variable' in adata.var.columns:
            hvg_col = 'vst.variable'
    if args.hvg_only and hvg_col is not None:
        hvg_mask = adata.var[hvg_col].values.astype(bool)
        gene_indices = np.where(hvg_mask)[0]
        if 'feature_name' in adata.var.columns:
            all_names = adata.var['feature_name'].values.astype(str)
        else:
            all_names = np.array(adata.var_names, dtype=str)
        gene_names = [all_names[i] for i in gene_indices]
        print(f"  HVG-only: {len(gene_names)} / {n_genes_total} genes")
    else:
        gene_indices = np.arange(n_genes_total)
        if 'feature_name' in adata.var.columns:
            gene_names = list(adata.var['feature_name'].values.astype(str))
        else:
            gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    # Expression strategy
    if use_backed:
        expr_source = 'h5ad'
        del adata
        gc.collect()
        print(f"{elapsed()} Backed adata closed. Expression will be read via h5py.")
    else:
        expr_source = 'sparse'
        # Get expression matrix (prefer .raw for full gene set)
        if adata.raw is not None and adata.raw.shape[1] > adata.shape[1]:
            print(f"  Using .raw expression ({adata.raw.shape[1]} genes)")
            expr_mat = adata.raw.X
            # Re-derive gene names from raw if not HVG-filtered
            if not args.hvg_only:
                gene_names = list(adata.raw.var_names)
                gene_indices = np.arange(len(gene_names))
                n_genes_total = len(gene_names)
                n_genes = len(gene_names)
        else:
            expr_mat = adata.X

        if issparse(expr_mat):
            expr_sparse = csc_matrix(expr_mat.T)  # genes x cells
        else:
            expr_sparse = csc_matrix(expr_mat.T)
        # Apply HVG filter if needed
        if args.hvg_only and len(gene_indices) < n_genes_total:
            expr_sparse = expr_sparse[gene_indices, :]
        print(f"  Expression (in-memory): {expr_sparse.shape[0]} genes x {expr_sparse.shape[1]} cells")
        del adata
        gc.collect()


def load_from_export(export_dir):
    """Load data from legacy R/Seurat export directory."""
    global X_emb, cell_types, unique_types, meta_col_names, metadata_cols
    global gene_names, gene_indices, n_cells, n_genes, n_genes_total
    global expr_source, expr_sparse
    import pandas as pd
    from scipy.sparse import csc_matrix

    print(f"{elapsed()} Loading from export directory: {export_dir}")

    pca_df = pd.read_csv(os.path.join(export_dir, "pca.csv"), index_col=0)
    X_emb = pca_df.values.astype(np.float32)
    if X_emb.shape[1] > 50:
        X_emb = X_emb[:, :50]
    print(f"  PCA: {X_emb.shape}")

    meta = pd.read_csv(os.path.join(export_dir, "metadata.csv"), index_col=0)

    if args.filter_region:
        mask = meta["region"].values == args.filter_region
        meta = meta[mask].copy()
        X_emb = X_emb[mask]
        print(f"  Filtered to region={args.filter_region}: {X_emb.shape[0]} cells")

    ct_col = args.cell_type_col or "cell_type"
    cell_types = meta[ct_col].values.astype(str)
    unique_types = sorted(set(cell_types))

    # Metadata columns
    if args.metadata_cols:
        meta_col_names = [c.strip() for c in args.metadata_cols.split(",")]
    else:
        meta_col_names = [ct_col, "seurat_clusters", "orig.ident", "region", "subgroup"]
        meta_col_names = [c for c in meta_col_names if c in meta.columns]

    for col in meta_col_names:
        if col in meta.columns:
            vals = meta[col].values.astype(str).tolist()
            uniq = sorted(set(vals))
            label = META_COL_LABELS.get(col, col.replace("_", " ").title())
            metadata_cols[col] = {"label": label, "values": vals, "unique": uniq}

    # Expression
    print("  Loading sparse expression...")
    gene_names_list = open(os.path.join(export_dir, "gene_names.txt")).read().strip().split("\n")
    dims = np.fromfile(os.path.join(export_dir, "expr_dim.bin"), dtype=np.int32)
    ng, nc = int(dims[0]), int(dims[1])
    expr_i = np.fromfile(os.path.join(export_dir, "expr_i.bin"), dtype=np.int32)
    expr_p = np.fromfile(os.path.join(export_dir, "expr_p.bin"), dtype=np.int32)
    expr_x = np.fromfile(os.path.join(export_dir, "expr_x.bin"), dtype=np.float32)
    sparse = csc_matrix((expr_x, expr_i, expr_p), shape=(ng, nc))

    if args.filter_region:
        cell_idx = np.where(mask)[0]
        sparse = sparse[:, cell_idx]

    gene_names = gene_names_list
    gene_indices = np.arange(len(gene_names))
    n_cells = X_emb.shape[0]
    n_genes = len(gene_names)
    n_genes_total = n_genes
    expr_source = 'sparse'
    expr_sparse = sparse
    print(f"  Expression: {expr_sparse.shape[0]} genes x {expr_sparse.shape[1]} cells")


# -- Dispatch loading -------------------------------------------------
if args.input and args.input.endswith(".h5ad"):
    load_from_h5ad(args.input)
elif args.export_dir:
    load_from_export(args.export_dir)
elif os.path.isdir(os.path.join(os.path.dirname(__file__), "export")):
    # Backward compat: look for export/ next to this script
    load_from_export(os.path.join(os.path.dirname(__file__), "export"))
else:
    raise ValueError("Provide --input (h5ad) or --export-dir (legacy R export)")

print(f"  Cells: {n_cells:,}, Genes: {n_genes:,}, Types: {len(unique_types)}")


# =========================================================================
#  UMAP
# =========================================================================

if args.skip_umap:
    frames_path = os.path.join(OUT_DIR, "frames.bin")
    frames_size = os.path.getsize(frames_path)
    n_frames = frames_size // (n_cells * 3 * 4)
    nn_labels = ["static"] * n_frames  # placeholder
    print(f"\n{elapsed()} Skipping UMAP (--skip-umap), reusing frames.bin ({n_frames} frames)")

elif args.sweep:
    # Parse "start-end-nkeys"
    parts = args.sweep.split("-")
    nn_start, nn_end, n_keys = int(parts[0]), int(parts[1]), int(parts[2])
    nn_values = np.unique(np.linspace(nn_start, nn_end, n_keys).astype(int))
    n_interp = args.n_interp_frames
    use_random_init = n_cells > LARGE_THRESHOLD

    print(f"\n{elapsed()} UMAP sweep: {len(nn_values)} keyframes, nn={nn_values[0]}..{nn_values[-1]}")
    print(f"  Keyframes: {nn_values.tolist()}")
    print(f"  Init: {'random' if use_random_init else 'spectral'}")

    import umap

    # Checkpoint directory for crash recovery
    ckpt_dir = os.path.join(OUT_DIR, "_keyframe_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    embeddings = {}
    for ki, nn in enumerate(nn_values):
        ckpt_path = os.path.join(ckpt_dir, f"keyframe_nn{nn}.npy")
        if os.path.exists(ckpt_path):
            print(f"\n{elapsed()} Keyframe {ki+1}/{len(nn_values)}: nn={nn} (checkpoint)")
            embeddings[nn] = np.load(ckpt_path)
            continue

        print(f"\n{elapsed()} Keyframe {ki+1}/{len(nn_values)}: nn={nn}")
        reducer = umap.UMAP(
            n_components=3, n_neighbors=int(nn), min_dist=0.3,
            random_state=42, low_memory=use_random_init,
            init='random' if use_random_init else 'spectral',
            verbose=True,
        )
        embeddings[nn] = reducer.fit_transform(X_emb)
        np.save(ckpt_path, embeddings[nn])
        print(f"  Checkpoint saved.")
        del reducer
        gc.collect()

    del X_emb
    gc.collect()

    # Procrustes align to middle keyframe
    ref_nn = nn_values[len(nn_values) // 2]
    print(f"\n{elapsed()} Aligning to nn={ref_nn}...")
    reference = embeddings[ref_nn]
    aligned = {}
    for nn in nn_values:
        aligned[nn] = reference.copy() if nn == ref_nn else procrustes_align(reference, embeddings[nn])
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
        print(f"  dim {dim+1}/3 done... {elapsed()}")
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
    use_random_init = n_cells > LARGE_THRESHOLD
    print(f"\n{elapsed()} Computing static 3D UMAP ({n_cells:,} cells, nn={args.umap_neighbors})")
    print(f"  Init: {'random' if use_random_init else 'spectral'}")

    import umap
    reducer = umap.UMAP(
        n_components=3, n_neighbors=args.umap_neighbors, min_dist=0.3,
        random_state=42, low_memory=use_random_init,
        init='random' if use_random_init else 'spectral',
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
    nn_labels = ["static"]

    frames_path = os.path.join(OUT_DIR, "frames.bin")
    frames.tofile(frames_path)
    print(f"{elapsed()} frames.bin: {os.path.getsize(frames_path) / 1024 / 1024:.1f} MB")
    del frames, coords_3d
    gc.collect()


# =========================================================================
#  COLORS
# =========================================================================

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
colors.tofile(os.path.join(OUT_DIR, "colors.bin"))
print(f"  colors.bin: {os.path.getsize(os.path.join(OUT_DIR, 'colors.bin')) / 1024 / 1024:.1f} MB")
del colors
gc.collect()


# =========================================================================
#  EXPRESSION
# =========================================================================

if args.skip_expression:
    print(f"\n{elapsed()} Skipping expression (--skip-expression)")
elif expr_source == 'sparse':
    # -- In-memory path (small datasets / legacy export) --
    print(f"\n{elapsed()} Writing expression (in-memory, {n_genes:,} genes)...")
    gene_index = {}
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    offset = 0
    batch_size = 500

    with open(os.path.join(OUT_DIR, "expression.bin"), "wb") as f:
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            batch = expr_sparse[start:end, :].toarray()
            for local_i, gi in enumerate(range(start, end)):
                col = batch[local_i]
                mn, mx = col.min(), col.max()
                gene_totals[gi] = col.sum()
                if mx > mn:
                    quantized = ((col - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    quantized = np.zeros(expr_sparse.shape[1], dtype=np.uint8)
                f.write(quantized.tobytes())
                gene_index[gene_names[gi]] = [offset, expr_sparse.shape[1]]
                offset += expr_sparse.shape[1]
            del batch
            print(f"  {end}/{n_genes} genes... {elapsed()}", flush=True)

    del expr_sparse
    gc.collect()

    expr_size = os.path.getsize(os.path.join(OUT_DIR, "expression.bin"))
    print(f"  expression.bin: {expr_size / 1024 / 1024:.1f} MB")

    with open(os.path.join(OUT_DIR, "gene_index.json"), "w") as f:
        json.dump(gene_index, f, separators=(",", ":"))
    print(f"  gene_index.json: {len(gene_index)} genes")

    ranked_indices = np.argsort(-gene_totals)
    gene_ranks = {gene_names[idx]: rank for rank, idx in enumerate(ranked_indices, 1)}
    with open(os.path.join(OUT_DIR, "gene_ranks.json"), "w") as f:
        json.dump(gene_ranks, f, separators=(",", ":"))
    print(f"  gene_ranks.json: {len(gene_ranks)} genes ranked")

elif expr_source == 'h5ad':
    # -- h5py chunked path (large datasets) --
    print(f"\n{elapsed()} Writing expression (h5py chunked, {n_genes:,} genes)...")

    from scipy.sparse import csr_matrix
    import h5py

    buf_gb = n_genes * n_cells * 4 / 1024**3
    print(f"  Allocating buffer: {n_genes} x {n_cells:,} float32 = {buf_gb:.1f} GB")
    expr_buf = np.zeros((n_genes, n_cells), dtype=np.float32)

    cell_chunk_size = 50000

    with h5py.File(h5ad_path, 'r') as f:
        X_grp = f['X']
        encoding = X_grp.attrs.get('encoding-type', b'csr_matrix')
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        print(f"  Sparse encoding: {encoding}")

        data_ds = X_grp['data']
        indices_ds = X_grp['indices']
        indptr_ds = X_grp['indptr']

        print(f"  Reading indptr ({indptr_ds.shape[0]:,} entries)...")
        indptr_full = indptr_ds[:]

        n_chunks = (n_cells + cell_chunk_size - 1) // cell_chunk_size
        print(f"  Processing {n_chunks} cell chunks of {cell_chunk_size:,}...")

        for ci, chunk_start in enumerate(range(0, n_cells, cell_chunk_size)):
            chunk_end = min(chunk_start + cell_chunk_size, n_cells)
            n_chunk_cells = chunk_end - chunk_start

            ptr_start = int(indptr_full[chunk_start])
            ptr_end = int(indptr_full[chunk_end])

            chunk_data = data_ds[ptr_start:ptr_end]
            chunk_indices = indices_ds[ptr_start:ptr_end]
            local_indptr = indptr_full[chunk_start:chunk_end + 1] - ptr_start

            chunk_csr = csr_matrix(
                (chunk_data, chunk_indices, local_indptr),
                shape=(n_chunk_cells, n_genes_total)
            )
            chunk_hvg = chunk_csr[:, gene_indices].toarray().astype(np.float32)
            expr_buf[:, chunk_start:chunk_end] = chunk_hvg.T

            del chunk_data, chunk_indices, chunk_csr, chunk_hvg
            gc.collect()

            if (ci + 1) % 5 == 0 or chunk_end == n_cells:
                print(f"    {chunk_end:,}/{n_cells:,} cells... {elapsed()}", flush=True)

    print(f"{elapsed()} Quantizing and writing expression.bin...")
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

    ranked_indices = np.argsort(-gene_totals)
    gene_ranks = {gene_names[idx]: rank for rank, idx in enumerate(ranked_indices, 1)}
    with open(os.path.join(OUT_DIR, "gene_ranks.json"), "w") as f:
        json.dump(gene_ranks, f, separators=(",", ":"))
    print(f"  gene_ranks.json: {len(gene_ranks)} genes ranked")


# =========================================================================
#  METADATA JSON
# =========================================================================

print(f"{elapsed()} Writing metadata.json...")

metadata = {
    "n_cells": n_cells,
    "n_frames": n_frames,
    "nn_labels": nn_labels,
    "cell_types": cell_types.tolist() if hasattr(cell_types, 'tolist') else list(cell_types),
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


# =========================================================================
#  SUMMARY
# =========================================================================

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
