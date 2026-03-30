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

EXPORT_DIR = "C:/Users/Nautilus/Desktop/scRNAseq/export"
OUT_DIR = "C:/Users/Nautilus/Desktop/scRNAseq/viewer"
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

# ── Write metadata.json ────────────────────────────────────
metadata = {
    "n_cells": n_cells,
    "n_frames": n_frames,
    "nn_labels": [f"{v:.0f}" for v in nn_fine_pp],
    "cell_types": cell_types.tolist(),
    "unique_types": unique_types,
    "palette": palette,
    "gene_names": gene_names,
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, separators=(",", ":"))
print(f"  metadata.json written")

# ── Write index.html ────────────────────────────────────────
print("Writing index.html...")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neural scRNAseq - UMAP Warp</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; font-family: 'Segoe UI', sans-serif; color: #fff; }
  canvas { display: block; }

  #loading {
    position: fixed; inset: 0; background: #000; display: flex;
    align-items: center; justify-content: center; z-index: 999;
    flex-direction: column; gap: 16px;
  }
  #loading .bar { width: 300px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; }
  #loading .fill { height: 100%; background: #888; border-radius: 2px; width: 0%; transition: width 0.3s; }
  #loading .msg { font-size: 13px; opacity: 0.5; }

  #overlay {
    position: absolute; top: 0; left: 0; width: 100%; padding: 16px 24px;
    pointer-events: none; z-index: 10;
  }
  #title { font-size: 20px; font-weight: 300; letter-spacing: 1px; margin-bottom: 4px; }
  #param-label { font-size: 14px; opacity: 0.6; }

  #panel {
    position: absolute; top: 16px; left: 24px; z-index: 10;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 16px 20px;
    min-width: 220px;
    max-height: calc(100vh - 80px);
    overflow-y: auto;
    pointer-events: auto;
  }
  #panel h3 {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; opacity: 0.4; margin-bottom: 12px;
  }
  .panel-section { margin-bottom: 16px; }
  .panel-section:last-child { margin-bottom: 0; }
  .panel-label {
    font-size: 11px; opacity: 0.5; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 6px;
  }
  .panel-slider { width: 100%; accent-color: #888; margin-top: 4px; }
  .slider-value { font-size: 12px; opacity: 0.6; text-align: right; margin-top: 2px; }

  .gene-input-wrap { position: relative; }
  .gene-input-wrap input {
    width: 100%; padding: 6px 8px; font-size: 13px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px; color: #fff; outline: none;
  }
  .gene-input-wrap input::placeholder { color: rgba(255,255,255,0.3); }
  .gene-input-wrap input:focus { border-color: rgba(255,255,255,0.4); }
  #gene-dropdown {
    position: absolute; top: 100%; left: 0; right: 0;
    background: rgba(20,20,20,0.95); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 0 0 4px 4px; max-height: 200px; overflow-y: auto;
    display: none; z-index: 100;
  }
  #gene-dropdown .gene-option { padding: 5px 8px; font-size: 12px; cursor: pointer; }
  #gene-dropdown .gene-option:hover { background: rgba(255,255,255,0.1); }
  #gene-dropdown .gene-option.highlighted { background: rgba(255,255,255,0.15); }
  .gene-mapping-select {
    width: 100%; margin-top: 6px; padding: 4px 6px; font-size: 12px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px; color: #fff; outline: none;
  }
  .gene-mapping-select option { background: #111; }
  .gene-status { font-size: 11px; opacity: 0.5; margin-top: 4px; min-height: 1em; }
  .btn-clear {
    font-size: 11px; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15); border-radius: 3px;
    color: #fff; padding: 2px 8px; cursor: pointer; margin-top: 4px;
  }

  #rec-indicator {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(200,30,30,0.85); padding: 8px 20px; border-radius: 6px;
    font-size: 14px; font-weight: 600; letter-spacing: 1px;
    display: none; z-index: 20;
    animation: rec-pulse 1s ease-in-out infinite;
  }
  @keyframes rec-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  #controls {
    position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: 12px; z-index: 10;
  }
  #controls button {
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    color: #fff; padding: 6px 16px; cursor: pointer; border-radius: 4px;
    font-size: 13px;
  }
  #controls button:hover { background: rgba(255,255,255,0.2); }
  #controls button.recording { background: rgba(200,30,30,0.4); border-color: rgba(200,30,30,0.6); }
  #slider { width: 500px; accent-color: #888; }
  #rec-resolution, #rec-format {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2);
    color: #fff; padding: 5px 8px; border-radius: 4px; font-size: 12px; outline: none;
  }
  #rec-resolution option, #rec-format option { background: #111; }

  #legend {
    position: absolute; top: 16px; right: 24px; z-index: 10;
    font-size: 11px; line-height: 1.6;
    max-height: calc(100vh - 80px); overflow-y: auto;
  }
  .legend-item {
    display: flex; align-items: center; gap: 6px;
    cursor: pointer; user-select: none; padding: 1px 0;
    transition: opacity 0.2s;
  }
  .legend-item:hover { opacity: 0.8; }
  .legend-item.disabled { opacity: 0.3; text-decoration: line-through; }
  .legend-dot {
    width: 8px; height: 8px; border-radius: 50%;
    box-shadow: 0 0 4px 1px currentColor; flex-shrink: 0;
  }
  .legend-item.disabled .legend-dot { box-shadow: none; }
</style>
</head>
<body>

<div id="loading">
  <div class="msg" id="load-msg">Loading data...</div>
  <div class="bar"><div class="fill" id="load-fill"></div></div>
</div>

<div id="overlay">
  <div id="title">Neural scRNAseq &mdash; Continuous UMAP Warp</div>
  <div id="param-label">n_neighbors ~ 5</div>
</div>

<div id="rec-indicator">REC</div>

<div id="panel" style="display:none">
  <h3>Controls</h3>

  <div class="panel-section">
    <div class="panel-label">Gene Expression</div>
    <div class="gene-input-wrap">
      <input type="text" id="gene-input" placeholder="Type gene symbol..." autocomplete="off">
      <div id="gene-dropdown"></div>
    </div>
    <select class="gene-mapping-select" id="gene-mapping">
      <option value="brightness">Map to: Brightness</option>
      <option value="size">Map to: Point Size</option>
    </select>
    <div class="gene-status" id="gene-status"></div>
    <button class="btn-clear" id="gene-clear">Clear gene</button>
  </div>

  <div class="panel-section">
    <div class="panel-label">Point Brightness</div>
    <input type="range" class="panel-slider" id="brightness" min="0.2" max="4.0" step="0.1" value="1.8">
    <div class="slider-value" id="brightness-val">1.8</div>
  </div>

  <div class="panel-section">
    <div class="panel-label">Bloom Strength</div>
    <input type="range" class="panel-slider" id="bloom-strength" min="0.0" max="4.0" step="0.1" value="1.5">
    <div class="slider-value" id="bloom-val">1.5</div>
  </div>

  <div class="panel-section">
    <div class="panel-label">Point Size</div>
    <input type="range" class="panel-slider" id="point-size" min="2" max="40" step="1" value="8">
    <div class="slider-value" id="size-val">8</div>
  </div>

  <div class="panel-section">
    <div class="panel-label">Trail Length</div>
    <input type="range" class="panel-slider" id="trail-length" min="0" max="0.98" step="0.01" value="0.0">
    <div class="slider-value" id="trail-val">Off</div>
  </div>
</div>

<div id="legend" style="display:none"></div>

<div id="controls" style="display:none">
  <button id="btn-play">Pause</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <button id="btn-speed" title="Cycle speed">1x</button>
  <select id="rec-resolution" title="Recording resolution">
    <option value="native">Native</option>
    <option value="nativesq">Native (square)</option>
    <option value="1080p">1080p</option>
    <option value="1080sq">1080p (square)</option>
    <option value="4k" selected>4K</option>
    <option value="4ksq">4K (square)</option>
  </select>
  <select id="rec-format" title="Recording format">
    <option value="webm">WebM</option>
    <option value="mp4">MP4</option>
  </select>
  <button id="btn-record">Record Loop</button>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ── Load external data ──
const loadFill = document.getElementById('load-fill');
const loadMsg = document.getElementById('load-msg');

function progress(pct, msg) {
  loadFill.style.width = pct + '%';
  loadMsg.textContent = msg;
}

progress(5, 'Loading metadata...');
const META = await (await fetch('metadata.json')).json();
const nCells = META.n_cells;
const nFrames = META.n_frames;

progress(15, 'Loading frames...');
const framesBuf = await (await fetch('frames.bin')).arrayBuffer();
const allFrames = new Float32Array(framesBuf);
// Split into per-frame arrays (each frame is nCells * 3 floats)
const frameSize = nCells * 3;
const frames = [];
for (let i = 0; i < nFrames; i++) {
  frames.push(allFrames.subarray(i * frameSize, (i + 1) * frameSize));
}

progress(60, 'Loading colors...');
const colorsBuf = await (await fetch('colors.bin')).arrayBuffer();
const baseColors = new Float32Array(colorsBuf);

progress(70, 'Loading gene index...');
const geneIndex = await (await fetch('gene_index.json')).json();

progress(80, 'Initializing viewer...');

// ── Scene ──
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 200);
camera.position.set(0, 0, 14);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.08;
orbitControls.rotateSpeed = 0.5;
orbitControls.enablePan = false;
orbitControls.target.set(0, 0, 0);
orbitControls.minDistance = 0.5;
orbitControls.maxDistance = 100;
orbitControls.zoomSpeed = 2.0;

// ── Bloom ──
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(innerWidth, innerHeight), 1.5, 0.4, 0.2
);
composer.addPass(bloom);
composer.renderToScreen = false;

// ── Trail accumulation ──
const rtParams = {
  minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat, type: THREE.HalfFloatType,
};
const pr = Math.min(devicePixelRatio, 2);
let trailA = new THREE.WebGLRenderTarget(innerWidth * pr, innerHeight * pr, rtParams);
let trailB = new THREE.WebGLRenderTarget(innerWidth * pr, innerHeight * pr, rtParams);

const fsQuadGeo = new THREE.PlaneGeometry(2, 2);
const fadeMaterial = new THREE.ShaderMaterial({
  uniforms: { tOld: { value: null }, tNew: { value: null }, uFade: { value: 0.0 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tOld; uniform sampler2D tNew; uniform float uFade;
    varying vec2 vUv;
    void main() {
      vec4 old = texture2D(tOld, vUv) * uFade;
      vec4 nw = texture2D(tNew, vUv);
      gl_FragColor = max(old, nw);
    }
  `,
});
const fsQuad = new THREE.Mesh(fsQuadGeo, fadeMaterial);
const fsScene = new THREE.Scene(); fsScene.add(fsQuad);
const fsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const outputMaterial = new THREE.ShaderMaterial({
  uniforms: { tDiffuse: { value: null } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; varying vec2 vUv; void main() { gl_FragColor = texture2D(tDiffuse, vUv); }`,
});
const outputQuad = new THREE.Mesh(fsQuadGeo.clone(), outputMaterial);
const outputScene = new THREE.Scene(); outputScene.add(outputQuad);
let trailFade = 0.0;

// ── Point cloud ──
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(nCells * 3);
const colorsArr = new Float32Array(nCells * 3);
const sizesArr = new Float32Array(nCells);
for (let i = 0; i < nCells * 3; i++) colorsArr[i] = baseColors[i];
sizesArr.fill(1.0);

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('aColor', new THREE.BufferAttribute(colorsArr, 3));
geometry.setAttribute('aSize', new THREE.BufferAttribute(sizesArr, 1));

let basePtSize = 8.0;

const material = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: {
    uSize: { value: basePtSize * pr },
    uBrightness: { value: 1.8 },
  },
  vertexShader: `
    attribute vec3 aColor;
    attribute float aSize;
    varying vec3 vColor;
    varying float vAlpha;
    uniform float uSize;
    void main() {
      vColor = aColor;
      vAlpha = aSize;
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = uSize * aSize / -mvPos.z;
      gl_Position = projectionMatrix * mvPos;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    varying float vAlpha;
    uniform float uBrightness;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float intensity = 1.0 - d * d;
      intensity = pow(intensity, 1.5);
      gl_FragColor = vec4(vColor * intensity * uBrightness, intensity * 0.85 * vAlpha);
    }
  `,
});

const points = new THREE.Points(geometry, material);
scene.add(points);

// ── Gene expression (fetched on demand) ──
let activeGene = null;
let activeGeneExpr = null;
let geneMapping = 'brightness';
const geneCache = {};  // gene -> Uint8Array

async function fetchGeneExpression(gene) {
  if (geneCache[gene]) return geneCache[gene];
  const info = geneIndex[gene];
  if (!info) return null;
  const [offset, length] = info;
  const resp = await fetch('expression.bin', {
    headers: { Range: `bytes=${offset}-${offset + length - 1}` }
  });
  const buf = await resp.arrayBuffer();
  const arr = new Uint8Array(buf);
  geneCache[gene] = arr;
  return arr;
}

function applyGeneOverlay() {
  const colors = geometry.attributes.aColor.array;
  const sizes = geometry.attributes.aSize.array;

  if (!activeGene || !activeGeneExpr) {
    for (let i = 0; i < nCells * 3; i++) colors[i] = baseColors[i];
    sizesArr.fill(1.0);
    for (let i = 0; i < nCells; i++) sizes[i] = 1.0;
  } else if (geneMapping === 'brightness') {
    for (let i = 0; i < nCells; i++) {
      const e = 0.08 + (activeGeneExpr[i] / 255) * 0.92;
      colors[i*3] = baseColors[i*3] * e;
      colors[i*3+1] = baseColors[i*3+1] * e;
      colors[i*3+2] = baseColors[i*3+2] * e;
      sizes[i] = 1.0;
    }
  } else {
    for (let i = 0; i < nCells * 3; i++) colors[i] = baseColors[i];
    for (let i = 0; i < nCells; i++) {
      sizes[i] = 0.15 + (activeGeneExpr[i] / 255) * 1.85;
    }
  }

  // Cluster visibility
  for (const ct of META.unique_types) {
    if (!clusterVisible[ct]) {
      for (const i of clusterCellIndices[ct]) {
        sizes[i] = 0;
        colors[i*3] = 0; colors[i*3+1] = 0; colors[i*3+2] = 0;
      }
    }
  }

  geometry.attributes.aColor.needsUpdate = true;
  geometry.attributes.aSize.needsUpdate = true;
}

async function selectGene(gene) {
  const status = document.getElementById('gene-status');
  status.textContent = 'Loading ' + gene + '...';
  const expr = await fetchGeneExpression(gene);
  if (!expr) { status.textContent = gene + ': not found'; return; }
  activeGene = gene;
  activeGeneExpr = expr;
  const nonzero = Array.from(expr).filter(v => v > 0).length;
  status.textContent = `${gene}: ${nonzero}/${nCells} cells expressing`;
  applyGeneOverlay();
}

function clearGene() {
  activeGene = null;
  activeGeneExpr = null;
  document.getElementById('gene-status').textContent = '';
  document.getElementById('gene-input').value = '';
  applyGeneOverlay();
}

// Gene autocomplete
const geneInput = document.getElementById('gene-input');
const geneDropdown = document.getElementById('gene-dropdown');
const allGeneNames = META.gene_names;
let highlightIdx = -1;

geneInput.addEventListener('input', () => {
  const q = geneInput.value.trim().toUpperCase();
  geneDropdown.innerHTML = '';
  highlightIdx = -1;
  if (q.length < 1) { geneDropdown.style.display = 'none'; return; }

  const startsWith = [];
  const contains = [];
  for (const g of allGeneNames) {
    const u = g.toUpperCase();
    if (u.startsWith(q)) startsWith.push(g);
    else if (u.includes(q) && contains.length < 10) contains.push(g);
    if (startsWith.length >= 20) break;
  }
  const matches = [...startsWith, ...contains];
  if (matches.length === 0) { geneDropdown.style.display = 'none'; return; }

  for (const g of matches) {
    const opt = document.createElement('div');
    opt.className = 'gene-option';
    opt.textContent = g;
    opt.addEventListener('mousedown', (e) => {
      e.preventDefault();
      geneInput.value = g;
      geneDropdown.style.display = 'none';
      selectGene(g);
    });
    geneDropdown.appendChild(opt);
  }
  geneDropdown.style.display = 'block';
});

geneInput.addEventListener('keydown', (e) => {
  const items = geneDropdown.querySelectorAll('.gene-option');
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    highlightIdx = Math.min(highlightIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('highlighted', i === highlightIdx));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    highlightIdx = Math.max(highlightIdx - 1, 0);
    items.forEach((el, i) => el.classList.toggle('highlighted', i === highlightIdx));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (highlightIdx >= 0 && items[highlightIdx]) items[highlightIdx].dispatchEvent(new MouseEvent('mousedown'));
    else if (items.length > 0) items[0].dispatchEvent(new MouseEvent('mousedown'));
  }
});
geneInput.addEventListener('blur', () => {
  setTimeout(() => { geneDropdown.style.display = 'none'; }, 200);
});
document.getElementById('gene-mapping').addEventListener('change', (e) => {
  geneMapping = e.target.value;
  applyGeneOverlay();
});
document.getElementById('gene-clear').addEventListener('click', clearGene);

// ── Legend with cluster toggle ──
const legendEl = document.getElementById('legend');
const clusterVisible = {};
const clusterCellIndices = {};
for (const ct of META.unique_types) {
  clusterVisible[ct] = true;
  clusterCellIndices[ct] = [];
}
for (let i = 0; i < nCells; i++) {
  clusterCellIndices[META.cell_types[i]].push(i);
}
for (const ct of META.unique_types) {
  const c = META.palette[ct] || [0.5, 0.5, 0.5];
  const hex = '#' + c.map(v => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  const n = clusterCellIndices[ct].length;
  const div = document.createElement('div');
  div.className = 'legend-item';
  div.innerHTML = `<span class="legend-dot" style="background:${hex};color:${hex}"></span>${ct} (${n})`;
  div.addEventListener('click', () => {
    clusterVisible[ct] = !clusterVisible[ct];
    div.classList.toggle('disabled', !clusterVisible[ct]);
    applyGeneOverlay();
  });
  legendEl.appendChild(div);
}

// ── Animation state ──
let currentFrame = 0;
let playing = true;
let speed = 1;
const speeds = [0.1, 0.25, 0.5, 1, 2, 4];
let speedIdx = 3;
let accumulator = 0;

const paramLabel = document.getElementById('param-label');
const slider = document.getElementById('slider');
slider.max = nFrames - 1;
const btnPlay = document.getElementById('btn-play');
const btnSpeed = document.getElementById('btn-speed');

function setFrame(f) {
  currentFrame = f;
  const src = frames[f];
  const pos = geometry.attributes.position.array;
  for (let i = 0; i < nCells * 3; i++) pos[i] = src[i];
  geometry.attributes.position.needsUpdate = true;
  paramLabel.textContent = `n_neighbors ~ ${META.nn_labels[f]}`;
  slider.value = f;
}

// Panel sliders
document.getElementById('brightness').addEventListener('input', (e) => {
  material.uniforms.uBrightness.value = parseFloat(e.target.value);
  document.getElementById('brightness-val').textContent = parseFloat(e.target.value).toFixed(1);
});
document.getElementById('bloom-strength').addEventListener('input', (e) => {
  bloom.strength = parseFloat(e.target.value);
  document.getElementById('bloom-val').textContent = bloom.strength.toFixed(1);
});
document.getElementById('point-size').addEventListener('input', (e) => {
  basePtSize = parseFloat(e.target.value);
  material.uniforms.uSize.value = basePtSize * pr;
  document.getElementById('size-val').textContent = basePtSize.toFixed(0);
});
document.getElementById('trail-length').addEventListener('input', (e) => {
  trailFade = parseFloat(e.target.value);
  fadeMaterial.uniforms.uFade.value = trailFade;
  document.getElementById('trail-val').textContent = trailFade < 0.01 ? 'Off' : trailFade.toFixed(2);
});

btnPlay.addEventListener('click', () => {
  playing = !playing;
  btnPlay.textContent = playing ? 'Pause' : 'Play';
});
btnSpeed.addEventListener('click', () => {
  speedIdx = (speedIdx + 1) % speeds.length;
  speed = speeds[speedIdx];
  btnSpeed.textContent = speed + 'x';
});
slider.addEventListener('input', () => { setFrame(parseInt(slider.value)); });

// ── Recording ──
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordStartFrame = 0;
let recordLoopCount = 0;
let savedState = null;

const btnRecord = document.getElementById('btn-record');
const recIndicator = document.getElementById('rec-indicator');

function getRecDims() {
  const sel = document.getElementById('rec-resolution').value;
  const s = Math.min(innerWidth, innerHeight);
  return {
    'native': [innerWidth, innerHeight],
    'nativesq': [s, s],
    '1080p': [1920, 1080],
    '1080sq': [1080, 1080],
    '4k': [3840, 2160],
    '4ksq': [2160, 2160],
  }[sel] || [innerWidth, innerHeight];
}

function startRecording() {
  const [w, h] = getRecDims();
  const fmt = document.getElementById('rec-format').value;

  savedState = {
    width: renderer.domElement.width, height: renderer.domElement.height,
    cssWidth: renderer.domElement.style.width, cssHeight: renderer.domElement.style.height,
    pixelRatio: renderer.getPixelRatio(),
    trailSize: [trailA.width, trailA.height],
  };

  renderer.setPixelRatio(1);
  renderer.setSize(w, h);
  composer.setSize(w, h);
  trailA.setSize(w, h); trailB.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.domElement.style.width = savedState.cssWidth;
  renderer.domElement.style.height = savedState.cssHeight;

  const stream = renderer.domElement.captureStream(0);
  let mimeType = fmt === 'mp4'
    ? (MediaRecorder.isTypeSupported('video/mp4;codecs=avc1.42E01E') ? 'video/mp4;codecs=avc1.42E01E' : 'video/mp4')
    : (MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm');

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 20000000 });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: mimeType });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `umap_warp.${fmt}`;
    a.click();
    restoreAfterRecording();
  };

  isRecording = true;
  recordStartFrame = currentFrame;
  recordLoopCount = 0;
  playing = true;
  btnPlay.textContent = 'Pause';
  mediaRecorder.start();
  btnRecord.textContent = 'Stop Rec';
  btnRecord.classList.add('recording');
  recIndicator.style.display = 'block';
}

function restoreAfterRecording() {
  if (!savedState) return;
  renderer.setPixelRatio(savedState.pixelRatio);
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  trailA.setSize(savedState.trailSize[0], savedState.trailSize[1]);
  trailB.setSize(savedState.trailSize[0], savedState.trailSize[1]);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  savedState = null;
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  mediaRecorder.stop();
  btnRecord.textContent = 'Record Loop';
  btnRecord.classList.remove('recording');
  recIndicator.style.display = 'none';
}

btnRecord.addEventListener('click', () => {
  if (isRecording) stopRecording();
  else startRecording();
});

// ── Init + show UI ──
setFrame(0);
progress(100, 'Ready!');
setTimeout(() => {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('panel').style.display = '';
  document.getElementById('legend').style.display = '';
  document.getElementById('controls').style.display = '';
}, 300);

// ── Render loop ──
let lastTime = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  if (playing) {
    accumulator += dt * speed * 30;
    while (accumulator >= 1) {
      accumulator -= 1;
      const prevFrame = currentFrame;
      currentFrame = (currentFrame + 1) % nFrames;
      setFrame(currentFrame);

      if (isRecording) {
        if (currentFrame === 0 && prevFrame === nFrames - 1) recordLoopCount++;
        if (recordLoopCount >= 1 && currentFrame >= recordStartFrame) {
          stopRecording();
          break;
        }
      }
    }
  }

  orbitControls.update();

  if (trailFade < 0.01) {
    composer.renderToScreen = true;
    composer.render();
  } else {
    composer.renderToScreen = false;
    composer.render();
    fadeMaterial.uniforms.tOld.value = trailA.texture;
    fadeMaterial.uniforms.tNew.value = composer.readBuffer.texture;
    renderer.setRenderTarget(trailB);
    renderer.render(fsScene, fsCamera);
    outputMaterial.uniforms.tDiffuse.value = trailB.texture;
    renderer.setRenderTarget(null);
    renderer.render(outputScene, fsCamera);
    const tmp = trailA; trailA = trailB; trailB = tmp;
  }

  if (isRecording) {
    const stream = renderer.domElement.captureStream(0);
    const track = stream.getVideoTracks()[0];
    if (track && track.requestFrame) track.requestFrame();
  }
}

requestAnimationFrame(animate);

window.addEventListener('resize', () => {
  if (isRecording) return;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  const pr = Math.min(devicePixelRatio, 2);
  trailA.setSize(innerWidth * pr, innerHeight * pr);
  trailB.setSize(innerWidth * pr, innerHeight * pr);
});
</script>
</body>
</html>"""

with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n=== Output ===")
for fn in os.listdir(OUT_DIR):
    fp = os.path.join(OUT_DIR, fn)
    sz = os.path.getsize(fp)
    print(f"  {fn}: {sz / 1024 / 1024:.1f} MB" if sz > 1024*1024 else f"  {fn}: {sz / 1024:.1f} KB")

print(f"\nTo view:")
print(f"  cd {OUT_DIR}")
print(f"  python -m http.server 8000")
print(f"  Open http://localhost:8000 in your browser")
