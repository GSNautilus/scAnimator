# export_seurat.R -- export a Seurat object for scripts/build_viewer.py
# ======================================================================
#
# Writes the files that `build_viewer.py --export-dir` reads:
#
#   pca.csv          cells x PCs (the UMAP input), first column = cell barcode
#   metadata.csv     cells x metadata columns, first column = cell barcode
#   gene_names.txt   one gene per line, in matrix row order
#   expr_dim.bin     int32 [n_genes, n_cells]
#   expr_i.bin       int32 row indices      } the normalized expression matrix
#   expr_p.bin       int32 column pointers  } (genes x cells, sparse, CSC)
#   expr_x.bin       float32 values         }
#
# Usage (in R, with your object loaded):
#   source("scripts/export_seurat.R")
#   export_for_scanimator(obj, "export",
#                         cell_type = "cell_type",          # column for the default colouring
#                         columns   = c("seurat_clusters", "orig.ident"))
#
# Then:
#   python scripts/build_viewer.py --export-dir export --output-dir datasets/mydata \
#       --name "My data" --sweep 5-150-11 --metadata-cols cell_type,seurat_clusters,orig.ident
#   python scripts/update_dataset_index.py
#
# Needs: Seurat (v4 or v5), Matrix. The object needs a PCA (RunPCA).

library(Seurat)
library(Matrix)

export_for_scanimator <- function(obj, out_dir = "export", cell_type = "cell_type",
                                  columns = c("seurat_clusters", "orig.ident"),
                                  assay = DefaultAssay(obj), reduction = "pca", n_pcs = 50) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  cells <- colnames(obj)

  # PCA embedding (the UMAP input)
  emb <- Embeddings(obj, reduction = reduction)
  emb <- emb[cells, seq_len(min(n_pcs, ncol(emb))), drop = FALSE]
  write.csv(emb, file.path(out_dir, "pca.csv"), quote = FALSE)

  # Metadata: the cell type column is written as "cell_type" (what build_viewer.py expects)
  md <- obj@meta.data[cells, , drop = FALSE]
  if (!cell_type %in% colnames(md)) stop("no metadata column '", cell_type, "'")
  keep <- unique(c(cell_type, intersect(columns, colnames(md))))
  out <- md[, keep, drop = FALSE]
  colnames(out)[colnames(out) == cell_type] <- "cell_type"
  write.csv(out, file.path(out_dir, "metadata.csv"), quote = TRUE)

  # Normalized expression ("data" layer / slot), genes x cells, as CSC
  m <- tryCatch(LayerData(obj, assay = assay, layer = "data"),
                error = function(e) GetAssayData(obj, assay = assay, slot = "data"))
  m <- as(m[, cells], "CsparseMatrix")
  writeLines(rownames(m), file.path(out_dir, "gene_names.txt"))
  writeBin(as.integer(dim(m)), file.path(out_dir, "expr_dim.bin"), size = 4)
  writeBin(as.integer(m@i), file.path(out_dir, "expr_i.bin"), size = 4)
  writeBin(as.integer(m@p), file.path(out_dir, "expr_p.bin"), size = 4)
  writeBin(as.numeric(m@x), file.path(out_dir, "expr_x.bin"), size = 4)

  message(sprintf("scAnimator export: %d cells, %d PCs, %d genes, %d metadata columns -> %s",
                  length(cells), ncol(emb), nrow(m), ncol(out), normalizePath(out_dir)))
  invisible(out_dir)
}
