#!/bin/bash

# Define the header to prepend
HEADER="
# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
"

# Check if a directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

# Check if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory not found: $DIRECTORY"
    exit 1
fi

# Find all .py files recursively and process them
find "$DIRECTORY" -type f -name "*.py" | while read -r file; do
    # Check if the file already has the header
    if ! grep -q "^# SynDisco: Automated experiment creation and execution using only LLM agents" "$file"; then
        # Prepend the header
        echo "$HEADER" | cat - "$file" > temp && mv temp "$file"
        echo "Header added to: $file"
    else
        echo "Header already exists in: $file"
    fi
done