#!/bin/bash

# Directory to search
SEARCH_DIR="/path/to/specific/directory"

# Loop through all .html and .txt files in the directory and its subdirectories
find "$SEARCH_DIR" -type f \( -iname "*.html" -o -iname "*.txt" \) | while read -r file; do
    # Search for corean characters in the file
    if grep -Pq "[\x{1100}-\x{11FF}\x{3130}-\x{318F}\x{A960}-\x{A97F}\x{AC00}-\x{D7AF}\x{D7B0}-\x{D7FF}]" "$file"; then
        # corean characters found, remove the file
        echo "Removing file with corean characters: $file"
        rm "$file"
    fi
done