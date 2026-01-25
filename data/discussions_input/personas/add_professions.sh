#!/usr/bin/env bash

INPUT_FILE="$1"
OUTPUT_FILE="${2:-output.json}"

# List of random professions
PROFESSIONS=(
  "Software Engineer"
  "Data Analyst"
  "Teacher"
  "Mechanical Engineer"
  "Doctor"
  "Nurse"
  "Journalist"
  "Marketing Specialist"
  "Economist"
  "Civil Servant"
  "Electrician"
  "Carpenter"
  "Product Manager"
  "Researcher"
  "Statistician"
  "Consultant"
  "Sales Representative"
)

random_profession() {
  echo "${PROFESSIONS[$RANDOM % ${#PROFESSIONS[@]}]}"
}

# Clear output file
> "$OUTPUT_FILE"

while IFS= read -r line; do
  if [[ "$line" =~ \"current_employment\"[[:space:]]*:[[:space:]]*\"Employed\" ]]; then
    prof=$(random_profession)
    echo "  \"current_employment\": \"$prof\"," >> "$OUTPUT_FILE"
  else
    echo "$line" >> "$OUTPUT_FILE"
  fi
done < "$INPUT_FILE"
