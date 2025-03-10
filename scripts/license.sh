#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <root_directory> "
    exit 1
fi

# Assign arguments to variables
ROOT_DIR="$1"

STRING_TO_PREPEND="
\"\"\"
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
\"\"\"

"

# Find all .py files excluding __init__.py and prepend the string
find "$ROOT_DIR" -type f -name "*.py" ! -name "__init__.py" | while read -r file; do
    # Create a temporary file to store the result
    tmp_file=$(mktemp)
    
    # Prepend the string and append the original content
    echo "$STRING_TO_PREPEND" > "$tmp_file"
    cat "$file" >> "$tmp_file"
    
    # Move the temporary file back to the original file
    mv "$tmp_file" "$file"
done

echo "String prepended successfully to all .py files (excluding __init__.py) in $ROOT_DIR"
