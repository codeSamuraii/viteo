#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VDEX_DIR="$(dirname "$SCRIPT_DIR")"

# Verify we're in the correct location
if [[ "$(basename "$VDEX_DIR")" != "viteo" ]] || [[ ! -d "$VDEX_DIR" ]]; then
    echo "Error: Rebuild script must be in 'viteo/build/'" >&2
    exit 1
fi

BUILD_DIR="$VDEX_DIR/build"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory not found" >&2
    exit 1
fi

# Clean build directory (except this script)
echo "* Cleaning build directory..."
CURRENT_SCRIPT="$(basename "$0")"
find "$BUILD_DIR" -mindepth 1 -maxdepth 1 ! -name "$CURRENT_SCRIPT" -exec rm -rf {} +

# Force rebuild using pip
echo "* Triggering rebuild..."
cd "$(dirname "$VDEX_DIR")"
if ! poetry run python -m pip install --force-reinstall --no-deps -e "$VDEX_DIR"; then
    echo "xxx Build failed" >&2
    exit 1
fi

# Verify the extension can be imported
echo " -> Verifying import..."
if poetry run python -c "import viteo; print(f' -> Version: {viteo.__version__}'); print(f' -> Module: {viteo.__file__}')"; then
    echo "*** Extension rebuilt successfully!"
else
    echo "xxx Could not import viteo" >&2
    exit 1
fi

echo " -> Cleaning build artifacts..."
KEEP_FILES=("rebuild.sh" "_viteo.cpython-313-darwin.so" "libnanobind-static.a")
for file in "$BUILD_DIR"/* "$BUILD_DIR"/.*; do
    # Skip if glob didn't match anything
    [[ -e "$file" ]] || continue
    filename="$(basename "$file")"
    # Skip current and parent directory entries
    [[ "$filename" == "." || "$filename" == ".." ]] && continue
    if [[ ! " ${KEEP_FILES[*]} " =~ " ${filename} " ]]; then
        rm -rf "$file"
    fi
done
