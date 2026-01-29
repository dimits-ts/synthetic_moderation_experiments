#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR="$1"

echo "Scanning parent directory: $PARENT_DIR"

for subdir in "$PARENT_DIR"/*/; do
  echo "Processing: $subdir"

  # Delete discussion.csv
  if [[ -f "${subdir}datasets/discussion.csv" ]]; then
    echo "  Deleting datasets/discussion.csv"
    rm -f "${subdir}datasets/discussion.csv"
  fi

  # Move JSON files from raw/ to subdir
  if [[ -d "${subdir}raw" ]]; then
    shopt -s nullglob
    json_files=("${subdir}raw"/*.json)
    shopt -u nullglob

    if (( ${#json_files[@]} > 0 )); then
      echo "  Moving ${#json_files[@]} JSON files"
      mv "${subdir}raw"/*.json "$subdir"
    fi
  fi
done

# Delete empty directories (bottom-up)
echo "Removing empty directories..."
find "$PARENT_DIR" -type d -empty -delete
