#!/bin/bash

# Directories
dir1="./discussions_output"
dir2="./annotation_output"

# Temp file to store matching IDs
tmp_ids=$(mktemp)

echo "🔍 Collecting IDs from $dir1..."

# Temporary file to store all valid IDs
temp_ids=$(mktemp)

# Recursively find all JSON files in dir1 and extract IDs
find "$dir1" -type f -name "*.json" | while read -r file; do
    id=$(jq -r '.id' "$file")
    if [[ -n "$id" && "$id" != "null" ]]; then
        echo "$id"
    fi
done > "$temp_ids"

id_count=$(wc -l < "$temp_ids")
echo "Found $id_count IDs in $dir1."

# Now recursively find all JSON files in dir2
find "$dir2" -type f -name "*.json" | while read -r file; do
    file_id=$(jq -r '.conv_id' "$file")
    
    if [[ -z "$file_id" || "$file_id" == "null" ]]; then
        echo "Deleting $file (no id found)"
        rm "$file"
    elif ! grep -Fxq "$file_id" "$temp_ids"; then
        echo "Deleting $file (id not found)"
        rm "$file"
    else
        echo "Keeping $file (id found)"
    fi
done

# Cleanup
rm "$temp_ids"