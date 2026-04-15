"""
update_dataset_index.py -- Scan datasets/ and write datasets/index.json.
========================================================================

The landing page (index.html) reads datasets/index.json to discover
available datasets. This script scans for subdirectories that contain a
metadata.json file and writes the index.

Usage:
  python scripts/update_dataset_index.py
  python scripts/update_dataset_index.py --datasets-dir /path/to/datasets
"""

import json
import os
import argparse

parser = argparse.ArgumentParser(description="Update datasets/index.json for scAnimator landing page")
parser.add_argument("--datasets-dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "datasets"),
                    help="Path to datasets/ directory")
args = parser.parse_args()

datasets_dir = args.datasets_dir

if not os.path.isdir(datasets_dir):
    print(f"Error: {datasets_dir} is not a directory")
    exit(1)

dataset_ids = []
for entry in sorted(os.listdir(datasets_dir)):
    if entry.startswith("_"):
        continue  # skip _local
    subdir = os.path.join(datasets_dir, entry)
    if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "metadata.json")):
        dataset_ids.append(entry)

index_path = os.path.join(datasets_dir, "index.json")
with open(index_path, "w") as f:
    json.dump(dataset_ids, f, indent=2)

print(f"Updated {index_path}: {len(dataset_ids)} datasets")
for ds in dataset_ids:
    meta_path = os.path.join(datasets_dir, ds, "metadata.json")
    size = os.path.getsize(meta_path)
    # Read just enough to get n_cells and name without loading the full 200MB file
    with open(meta_path, "r") as mf:
        head = mf.read(1000)
    # Quick parse for n_cells
    import re
    m = re.search(r'"n_cells":(\d+)', head)
    n_cells = int(m.group(1)) if m else "?"
    m = re.search(r'"name":"([^"]*)"', head)
    name = m.group(1) if m else ds
    print(f"  {ds}: {name} ({n_cells:,} cells)" if isinstance(n_cells, int) else f"  {ds}: {name}")
