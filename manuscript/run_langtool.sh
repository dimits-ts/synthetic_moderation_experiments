#!/bin/bash

# Color definitions
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
GREEN="\033[0;32m"
NC="\033[0m" # No Color

echo -e "${BLUE}Starting LanguageTool checks on .tex files (recursive)...${NC}"

# Check if jq is installed
if ! command -v jq &>/dev/null; then
  echo -e "${RED}Error: jq is not installed. Please install jq to parse LanguageTool output.${NC}"
  exit 1
fi

# Recursively find .tex files
find . -type f -name "*.tex" | while read -r file; do
  echo -e "\n${GREEN}Checking file: $file${NC}"

  # Clean content:
  # 1. Remove LaTeX comments (full-line and inline)
  # 2. Remove LaTeX commands
  # 3. Remove empty lines
  clean_text=$(sed -E 's/%.*//g' "$file" | \
               sed -E 's/\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?//g' | \
               sed '/^\s*$/d')

  if [[ -z "$clean_text" ]]; then
    echo -e "${YELLOW}Skipped (empty or LaTeX only)${NC}"
    continue
  fi

  # Call LanguageTool API
  response=$(curl -s -d "language=en-US" --data-urlencode "text=$clean_text" http://localhost:8081/v2/check)

  # Parse and display errors
  echo "$response" | jq -r '
    if .matches == [] then
      "✓ No issues found."
    else
      .matches[] |
      .context.text + "\n" +
      "│ \u001b[1;31mIssue:\u001b[0m " + .message + "\n" +
      "│ \u001b[1;33mSuggestion:\u001b[0m " + (.replacements[0].value // "None") + "\n" +
      "│ \u001b[0;34mRule:\u001b[0m " + .rule.id + " - " + .rule.description + "\n" +
      "─"
    end
  '
done

echo -e "\n${BLUE}All files checked.${NC}"

