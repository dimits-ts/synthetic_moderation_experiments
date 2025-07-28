#!/bin/bash

# Set source and target directories
SRC_DIR="./graphs"
DEST_DIR="./manuscript/resources"

# Loop over files in source directory
for file in "$SRC_DIR"/*; do
    filename=$(basename "$file")
    output_file="$DEST_DIR/$filename"

    # Use magick to convert RGB to CMYK
    magick "$file" -colorspace CMYK "$output_file"

    echo "Converted and copied: $filename"
done