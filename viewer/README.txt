scAnimator - Quick Start
========================

1. Extract the zip so the folder structure is preserved:
   index.html (landing page) + datasets/ folder with subdirectories

2. Open a terminal in the top-level folder and start a local server:

   Python:   python -m http.server 8000
   Node:     npx serve .
   PHP:      php -S localhost:8000

3. Open http://localhost:8000 in a browser (Firefox or Chrome recommended)

4. Select a dataset from the landing page to launch the viewer.

Note: Opening index.html directly (file://) will not work because
the viewer uses fetch() to load data files, which requires HTTP.

GitHub is maintained here: https://github.com/GSNautilus/scAnimator
