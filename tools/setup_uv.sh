#!/bin/bash

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

VENV_NAME=".venv"
PYTHON_VERSION="3.11"

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--name)
            VENV_NAME="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-n|--name <venv_name>] [-p|--python <python_version>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-n|--name <venv_name>] [-p|--python <python_version>]"
            exit 1
            ;;
    esac
done

# If venv doesn't exist, create it
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating $VENV_NAME with Python $PYTHON_VERSION..."
    uv venv -p "$PYTHON_VERSION" "$VENV_NAME"
else
    echo "$VENV_NAME already exists"
fi

# Activate the virtual environment
echo "Activating $VENV_NAME..."
. "$VENV_NAME/bin/activate"


uv pip install torch==2.6.0 torchaudio==2.6.0
uv pip install -e ../

# create activate_python.sh
echo "Creating activate_python.sh..."
SCRIPT_DIR=$(pwd)

cat > activate_python.sh << EOF
#!/bin/bash
. "$SCRIPT_DIR/$VENV_NAME/bin/activate"
EOF

chmod +x activate_python.sh
echo "Setup completed."
