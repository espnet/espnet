#!/bin/bash

set -euo pipefail

# Default values
PYTHON_VERSION="3.11"
TORCH_VERSION="2.9.1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --torch)
            TORCH_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --python VERSION   Python version (default: ${PYTHON_VERSION})"
            echo "  --torch VERSION    PyTorch version (default: ${TORCH_VERSION})"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Using Python version: ${PYTHON_VERSION}"
echo "Using PyTorch version: ${TORCH_VERSION}"

# Check pixi
if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi not found. Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
else
    echo "pixi is already installed"
fi

# Check uv
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
else
    echo "uv is already installed"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    pixi global install ffmpeg
fi

# If .venv doesn't exist, create it
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv -p "${PYTHON_VERSION}"
else
    echo ".venv already exists"
fi

# Activate the virtual environment
echo "Activating .venv..."
. .venv/bin/activate

uv pip install torch=="${TORCH_VERSION}" torchaudio=="${TORCH_VERSION}"
uv pip install -e ../

# create activate_python.sh
echo "Creating activate_python.sh..."
SCRIPT_DIR=$(pwd)

cat > activate_python.sh << EOF
#!/bin/bash
. "${SCRIPT_DIR}/.venv/bin/activate"
EOF

chmod +x activate_python.sh
echo "Setup completed."
