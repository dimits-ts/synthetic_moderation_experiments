#!/bin/bash

# Directory to start search
ROOT_DIR="$1"

if [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

# Find all JSON files
find "$ROOT_DIR" -type f -name "*.json" | while read -r file; do
  # Check if 'logs' exists and is an array
  if jq -e '.logs | arrays' "$file" > /dev/null 2>&1; then
    # Get the number of elements in 'logs'
    count=$(jq '.logs | length' "$file")
    
    if [ "$count" -gt 29 ]; then
      echo "Trimming $file (logs length: $count)"

      # Trim 'logs' to first 15 elements and overwrite the file
      jq '.logs |= .[:15]' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
  fi
done
