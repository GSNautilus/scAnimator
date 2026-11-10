<p align="center">
  <a href="https://gsnautilus.github.io/scAnimator/"><img src="media/banner.png" width="100%" alt="scAnimator: a 3D UMAP at n_neighbors 5, 10, 20 and 40"></a>
</p>

<p align="center">
  <img src="media/warp.gif" width="400" alt="A single-cell UMAP warping from n_neighbors 5 to 30 and back while the camera turns">
</p>

<h1 align="center">scAnimator</h1>

<p align="center">Rich 3D viewer and animator for scRNA-seq datasets.</p>

<p align="center">
  <a href="https://gsnautilus.github.io/scAnimator/"><b>Live demo</b></a> (PBMC 3k, runs in the browser) &middot;
  <a href="manual/scAnimator_Manual.pdf">User manual (PDF, 17 pages)</a> &middot;
  <a href="https://github.com/GSNautilus/scAnimator/releases/download/v1.5.0/scAnimator_tour.mp4">Video tour (MP4, 2 min)</a>
</p>

---

scAnimator is a 3D viewer and animator for single-cell RNA-seq data that runs in the
browser. It animates the UMAP as `n_neighbors` changes, so you can see how clusters hold
together or split. You can colour and filter cells by any metadata, overlay any gene, record
video up to 4K, and script a captioned walkthrough of a dataset. It works on datasets from a
few thousand cells to a million, from AnnData or Seurat.

## What it does

- Animation: 3D UMAPs at a series of `n_neighbors` values (for example 11 values from 5 to 150), aligned to each other (Procrustes) and joined with cubic splines into one loop. You can play, scrub, change speed or hold one frame.
- Colour and filter: colour by any categorical metadata column, a solid colour, or a colormap that cycles over time. Filter cells by any column, hide categories from the legend, and downsample groups to equal sizes.
- Gene expression: type a gene name (suggestions are ordered by total expression) and show it as colour (7 colormaps), point size or brightness, on a linear, log or quantile scale. Each gene is downloaded only when you ask for it.
- Rendering: points drawn with custom Three.js shaders, with bloom, persistence trails, depth of field and per-cell tracers.
- Camera: orbit and zoom with the mouse, or turn automatically. The animation and the camera turn share one timeline, so a loop ends where it started.
- Recording: renders the timeline frame by frame at 24 to 120 fps, at window size, 2K or 4K, as a ZIP of PNG frames. A square crop and the legend are optional.
- Storyboards: a JSON file lists steps (colour, filter, gene, `n_neighbors`, caption, or a text slide). The viewer plays them with transitions, stepping with the arrow keys, or `scripts/render_storyboard.py` renders them to MP4.
- Experimental: audio input (bass, mids and highs can change point size, brightness, colour or position) and a mirrored "fractal" view.
- Your own data: `scripts/build_viewer.py` converts an AnnData `.h5ad` file or a Seurat export into a dataset. Open it from a local server, or from disk with **Load Folder** (the files stay in the browser).

## A look around

<p align="center"><img src="media/interface.png" width="100%" alt="The viewer: control panel, a 39,000-cell dataset, legend and the control bar"></p>
<p align="center"><i>The viewer on a ~39,000-cell neural dataset, with the Render Settings group open.</i></p>

| | | |
|:---:|:---:|:---:|
| <img src="manual/img/sb_1_overview.png" alt="PBMC 3k by cell type"> | <img src="manual/img/sb_4_nkg7.png" alt="NKG7 expression"> | <img src="manual/img/fx_tracers.png" alt="Tracers"> |
| Colour by cell type (a storyboard frame) | A gene as a colormap | Tracers while the camera turns |
| <img src="manual/img/fx_persistence.png" alt="Persistence"> | <img src="manual/img/fx_bloom.png" alt="Bloom"> | <img src="manual/img/fx_fractal.png" alt="Fractal reflection"> |
| Persistence | Bloom | Fractal reflection |

These pictures were taken from the viewer by [`manual/capture.py`](manual/capture.py).
The [user manual](manual/scAnimator_Manual.pdf) covers every control.

## How it works

```
Seurat (R) ── scripts/export_seurat.R ──┐
                                        ├──> scripts/build_viewer.py ──> datasets/<id>/ ──> viewer.html (WebGL)
AnnData (.h5ad, Python) ────────────────┘
```

1. `build_viewer.py` runs a 3D UMAP at each `n_neighbors` value of the sweep, all from the same PCA (or any `obsm` embedding).
2. Each keyframe is Procrustes-aligned (rotation, reflection, scale) to the one before it, so rotations and flips between UMAP runs don't show up as motion.
3. A cubic spline gives each cell a path through the keyframes (180 frames by default). Playback runs up the sweep and back down.
4. Each gene is quantized to 8 bits and stored back to back in one file with a byte index, so the viewer fetches exactly one gene at a time with an HTTP Range request.
5. The viewer (`viewer.html`, one file, no build step) loads the frames as one binary buffer and draws them with Three.js: a custom `ShaderMaterial` with per-point colour and size, additive blending, `UnrealBloomPass`, render-target ping-pong for trails, a depth pass for depth of field.

Above 200,000 cells the build reads the `.h5ad` in backed mode and streams expression through `h5py`,
and a sweep keeps checkpoints so an interrupted run resumes.

## Bring your own data

From Python (AnnData):

```
pip install numpy pandas scipy scanpy umap-learn h5py
python scripts/build_viewer.py --input my_data.h5ad --output-dir datasets/my_data --name "My data" --sweep 5-150-11
python scripts/update_dataset_index.py
```

From R (Seurat v4 or v5):

```r
source("scripts/export_seurat.R")
export_for_scanimator(obj, "export", cell_type = "cell_type", columns = c("seurat_clusters", "orig.ident"))
```

```
python scripts/build_viewer.py --export-dir export --output-dir datasets/my_data --name "My data" --sweep 5-150-11 --metadata-cols cell_type,seurat_clusters,orig.ident
python scripts/update_dataset_index.py
```

Then serve the folder and open it:

```
python -m http.server 8000
```

Open <http://localhost:8000>. Or skip the server: on the [live demo](https://gsnautilus.github.io/scAnimator/)'s
landing page, **Load Folder** opens a built dataset straight from disk. The files are copied into your
browser's own storage and are never uploaded. `python scripts/build_viewer.py --help` lists every option;
the manual explains them.

## Storyboards

```json
{
  "dataset": "pbmc3k",
  "render": { "point_size": 40, "brightness": 0.8, "bloom": 0.5 },
  "steps": [
    { "name": "overview", "color_by": "cell_type", "n_neighbors": 5, "caption": "PBMC 3k" },
    { "name": "lymphocytes", "filter": { "cell_type": ["CD4 T", "CD8 T", "NK", "B"] }, "caption": "Lymphocytes" },
    { "name": "nkg7", "color_by": "gene:NKG7", "colormap": "magma", "caption": "NKG7" }
  ]
}
```

Drop the file on the viewer (or use the film-strip button) and step through it with the arrow keys; or
render it:

```
pip install playwright imageio-ffmpeg
python scripts/render_storyboard.py examples/pbmc3k_tour.json
```

A full example with a slide is in [`examples/pbmc3k_tour.json`](examples/pbmc3k_tour.json).

## Repository

```
index.html                    landing page: dataset cards, Load Folder
viewer.html                   the viewer (single file, Three.js from a CDN)
datasets/pbmc3k/              the public demo dataset
scripts/build_viewer.py       .h5ad or Seurat export -> dataset files
scripts/export_seurat.R       Seurat -> export for build_viewer.py
scripts/update_dataset_index.py
scripts/render_storyboard.py  storyboard -> MP4 clips
examples/pbmc3k_tour.json     an example storyboard
manual/                       the user manual: scanimator.html + style.css, capture.py (pictures), build.py (PDF)
```

## Demo data credit

The live demo uses the **3k PBMCs from a Healthy Donor** dataset from
[10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It was downloaded with
`scanpy.datasets.pbmc3k()` and modified: QC-filtered, normalized, annotated with canonical marker
genes, embedded with UMAP and quantized into the files in `datasets/pbmc3k/`. 10x Genomics does not
endorse this project.

The larger dataset in some pictures is a ~39,000-cell neural dataset; only pictures of it are
published here, not the data.

## License

The code is under the [MIT License](LICENSE). The PBMC 3k demo data in `datasets/pbmc3k/` stays under
CC BY 4.0 (see above).
