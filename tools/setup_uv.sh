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

# If .venv doesn't exist, create it
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv -p 3.11
else
    echo ".venv already exists"
fi

# Activate the virtual environment
echo "Activating .venv..."
. .venv/bin/activate


uv pip install torch==2.6.0 torchaudio==2.6.0
uv pip install -e ../

# create activate_python.sh
echo "Creating activate_python.sh..."
SCRIPT_DIR=$(pwd)

cat > activate_python.sh << EOF
#!/bin/bash
. "$SCRIPT_DIR/.venv/bin/activate"
EOF

chmod +x activate_python.sh
echo "Setup completed."
