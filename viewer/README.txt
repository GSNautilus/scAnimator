scAnimator - Quick Start
========================

1. Extract the zip so all files are in one folder:
   index.html, metadata.json, frames.bin, colors.bin,
   gene_index.json, gene_ranks.json, expression.bin (full version only)

2. Open a terminal in that folder and start a local server:

   Python:   python -m http.server 8000
   Node:     npx serve .
   PHP:      php -S localhost:8000

3. Open http://localhost:8000 in a browser (Firefox or Chrome recommended)

Note: Opening index.html directly (file://) will not work because
the viewer uses fetch() to load data files, which requires HTTP.

GitHub is maintained here: https://github.com/GSNautilus/scAnimator
