#!/usr/bin/env bash

# The dataset is a multi-speaker dataset, you can visit https://github.com/AI-Hobbyist/Genshin_Datasets?tab=readme-ov-file to get character packs.

# Check if 7z is installed
if ! command -v 7z &> /dev/null; then
    echo "Error: 7z is not installed. Please install p7zip-full first."
    echo "You can install it using: sudo apt-get install p7zip-full"
    exit 1
fi

# Check if LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Function to show usage
show_usage() {
    echo "Usage: $0 <download_dir> [languages...]"
    echo "Languages available: CN, JP, EN, KR"
    echo "Examples:"
    echo "  $0 /path/to/dir              # Interactive mode"
    echo "  $0 /path/to/dir CN JP        # Download CN and JP"
    echo "  $0 /path/to/dir all          # Download all languages"
    exit 1
}

# Check minimum arguments
if [ $# -lt 1 ]; then
    show_usage
fi

download_dir=$1
shift  # Remove the first argument (download_dir) from $@

# Define available languages and their URLs
declare -A urls=(
    ["CN"]="https://modelscope.cn/datasets/aihobbyist/Genshin_Dataset/resolve/master/Genshin5.1_CN.7z"
    ["JP"]="https://modelscope.cn/datasets/aihobbyist/Genshin_Dataset/resolve/master/Genshin5.1_JP.7z"
    ["EN"]="https://modelscope.cn/datasets/aihobbyist/Genshin_Dataset/resolve/master/Genshin5.1_EN.7z"
    ["KR"]="https://modelscope.cn/datasets/aihobbyist/Genshin_Dataset/resolve/master/Genshin5.1_KR.7z"
)

# Function to download and extract
download_and_extract() {
    local lang=$1
    local file_name="Genshin5.1_${lang}.7z"
    
    if [ -d "${download_dir}/Genshin5.1-${lang}" ]; then
        echo "Directory Genshin5.1-${lang} already exists. Skipping..."
        return 0
    fi
    
    echo "Downloading ${lang} dataset..."
    if wget -c "${urls[$lang]}" -O "${download_dir}/${file_name}"; then
        echo "Successfully downloaded ${file_name}"
        
        echo "Extracting ${file_name}..."
        cd "${download_dir}"
        if 7za x "${file_name} -mmt256"; then
            echo "Successfully extracted ${file_name}"
            case $lang in
                CN) mv "'中文'""Genshin5.1-${lang}";;
                EN) mv "'英语'""Genshin5.1-${lang}";;
                JP) mv "'日语'""Genshin5.1-${lang}";;
                KR) mv "'韩语'""Genshin5.1-${lang}";;
                *) echo "Unknown language code: $lang" ;;
            esac
            # rm "${file_name}"
        else
            echo "Failed to extract ${file_name}"
            return 1
        fi
    else
        echo "Failed to download ${file_name}"
        return 1
    fi
}

# Create download directory if it doesn't exist
mkdir -p "${download_dir}"

# If no languages specified, enter interactive mode
if [ $# -eq 0 ]; then
    echo "Please select languages to download (space-separated):"
    echo "Available options: CN JP EN KR all"
    read -r selected_langs
    set -- $selected_langs  # Convert input string to positional parameters
fi

# Process language selection
selected_languages=()
if [ "$1" = "all" ]; then
    selected_languages=("CN" "JP" "EN" "KR")
else
    for lang in "$@"; do
        if [[ ! " CN JP EN KR " =~ " $lang " ]]; then
            echo "Error: Invalid language code: $lang"
            show_usage
        fi
        selected_languages+=("$lang")
    done
fi

# Download and extract selected languages
for lang in "${selected_languages[@]}"; do
    download_and_extract "$lang"
done

echo "All selected downloads and extractions completed successfully."
