<p align="center">
  <img src="media/banner.png" width="720" alt="scAnimator rendering of a single-cell UMAP as glowing points on black">
</p>

<h1 align="center">scAnimator</h1>

<p align="center">
  <b>Watch single-cell data move.</b> A browser-based 3D viewer that animates a UMAP embedding<br>
  as it warps across <code>n_neighbors</code>, rendered as glowing points of light.
</p>

<p align="center">
  <a href="https://gsnautilus.github.io/scAnimator/"><b>▶&nbsp;Open the live demo</b></a>
  &nbsp;·&nbsp; PBMC 3k, runs in any WebGL2 browser, nothing to install
  <br><a href="https://github.com/GSNautilus/scAnimator/releases/download/v1.4.0/umap_warp_demo_720p.webm">Download a demo video</a> (WebM, 48&nbsp;s, 10&nbsp;MB)
</p>

---

UMAP plots change shape with their parameters, and a single static plot hides that. scAnimator
computes 3D UMAPs at a sweep of `n_neighbors` values, aligns them to each other, and interpolates
between them, so you can watch clusters merge, split and drift as the embedding moves from local
to global structure. On top of that animation you can color cells by metadata or gene expression,
filter populations, and record the result as a video.

<p align="center">
  <img src="media/interface.png" width="720" alt="Viewer interface: control panel, animated point cloud and legend">
  <br><sub>The viewer on a larger (~45k cell) neural dataset, with an earlier version of the interface.</sub>
</p>

## Features

- **Continuous UMAP warping.** Keyframes at a sweep of `n_neighbors` values (5–150 for the demo) are Procrustes-aligned and cubic-spline interpolated, so the transitions are smooth and the motion is meaningful.
- **Glow rendering.** Custom Three.js point shaders with additive blending, bloom, depth of field and optional motion trails.
- **Color and filter by metadata.** Any categorical column the build includes (cell type, cluster, sample, …). Click a legend entry to hide or show that category.
- **Gene expression overlay.** Type a gene symbol (autocomplete is ranked by total expression) and map it to a colormap (viridis, magma, plasma, inferno), point brightness or point size. Each gene is fetched on demand with an HTTP Range request, so the full matrix is never downloaded.
- **Normalize cell counts.** Downsample so every group of a chosen column shows the same number of cells.
- **360° rotation and a master timeline.** Morph and rotation loop together on one timeline with its own speed controls.
- **Recording.** Export loops as WebM or as a PNG frame sequence (zip) at a chosen FPS and resolution, with an optional square crop and legend overlay.
- **Experimental.** Audio reactivity, where points pulse and displace in time with a live audio input, and a fractal reflection mode.
- **Bring your own data.** Build any `.h5ad` file into a dataset, then open it from the landing page with **Load Folder**. The files stay in your browser and nothing is uploaded.

## Run it locally

The viewer is two static HTML files with no build step. Three.js is loaded from a CDN. Browsers block
`fetch` from `file://`, so serve the folder over HTTP:

```bash
python -m http.server 8000
# then open http://localhost:8000
```

## Build your own dataset

`scripts/build_viewer.py` turns an AnnData `.h5ad` file into the binary files the viewer reads.

```bash
pip install numpy pandas scipy scanpy umap-learn h5py

# Animated n_neighbors sweep: 11 keyframes from 5 to 150
python scripts/build_viewer.py --input my_data.h5ad --output-dir datasets/my_data \
    --name "My dataset" --sweep 5-150-11

# Register it on the landing page
python scripts/update_dataset_index.py
```

Useful options:

| Option | What it does |
| --- | --- |
| `--embedding-key X_pca` | `obsm` key used as UMAP input. If no embedding of 3 or more dimensions exists, PCA is computed. |
| `--cell-type-col` | `.obs` column for the default coloring (auto-detected if omitted) |
| `--metadata-cols a,b,c` | Which `.obs` columns to expose (default: all categorical) |
| `--sweep start-end-nkeys` | Animated sweep. Without it, a single static UMAP is built (fast). |
| `--hvg-only` | Store only highly variable genes, which keeps `expression.bin` small |
| `--skip-umap` / `--skip-expression` | Reuse existing outputs when rebuilding |

Datasets above 200k cells are read in backed mode, and expression is streamed through `h5py`, so large atlases
fit in memory. Run `python scripts/build_viewer.py --help` for every option.

## How it works

```
index.html        Landing page: lists datasets/index.json, or loads a local folder via IndexedDB
viewer.html       The viewer (single file, Three.js ES module)
datasets/<id>/    One folder per dataset
  frames.bin        Float32  nFrames × nCells × 3   all animation frames
  colors.bin        Float32  nCells × 3             default palette
  metadata.json     cell metadata, palettes, n_neighbors labels, gene names
  expression.bin    Uint8    per-gene quantized expression, concatenated
  gene_index.json   gene → [byte offset, length] for Range requests
  gene_ranks.json   gene → rank by total expression (autocomplete order)
scripts/
  build_viewer.py          .h5ad → dataset files
  update_dataset_index.py  rewrites datasets/index.json
```

- **Embedding sweep.** UMAP is run in 3D at each `n_neighbors` keyframe. Every keyframe is Procrustes-aligned to the previous one (rotation, reflection and scale), so only real structural change shows up as motion. A cubic spline fills in the frames between keyframes, and playback ping-pongs.
- **Shading.** A `ShaderMaterial` with per-point color and size attributes and a radial falloff of `pow(1 − d², 1.5)`, drawn with additive blending and then `UnrealBloomPass`.
- **Trails.** Render-target ping-pong: the previous frame is faded and combined with the new one using `max(old, new)`.

## Demo data credit

The live demo uses the **3k PBMCs from a Healthy Donor** dataset from
[10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It was downloaded with
`scanpy.datasets.pbmc3k()` and modified: QC-filtered, normalized, annotated with canonical
marker genes, embedded with UMAP and quantized into the files in `datasets/pbmc3k/`.
10x Genomics does not endorse this project.

## License

The code is under the [MIT License](LICENSE). The PBMC 3k demo data in `datasets/pbmc3k/` stays
under CC BY 4.0 (see above).
