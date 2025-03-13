
base_dir="/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/conf/template"

for path in "$base_dir"/*.yaml; do
    filename=$(basename "$path")  # Extract filename from full path
    prefix="${filename%%_*}"      # Extract prefix before the first '_'
    target_dir="$base_dir/$prefix"

    # Create directory if it doesn't exist
    mkdir -p "$target_dir"

    # Move file to the corresponding directory
    cp "$path" "$target_dir/${filename#*_}"
done