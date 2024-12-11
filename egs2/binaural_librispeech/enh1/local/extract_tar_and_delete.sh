#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/data_directory"
    exit 1
fi

# Define the directory containing the .tar.gz files from the input argument
DATA_DIR="$1"

# Find and loop through each .tar.gz file in the directory and its subdirectories
find "$DATA_DIR" -type f -name "*.tar.gz" | while read -r file; do
    echo "Extracting $file..."

    # Extract the .tar.gz file to the same directory as the archive
    tar -xzf "$file" -C "$(dirname "$file")" --exclude='._*'

    # Delete the .tar.gz file after extraction
    rm "$file"

    echo "$file extracted and removed."
done
