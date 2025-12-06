#!/bin/bash
# Helper script to use Python 3.11 for scripts that require giotto-tda

echo "Using Python 3.11 (with giotto-tda installed)"
echo ""

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is not installed."
    echo "Please install it with: brew install python@3.11"
    exit 1
fi

# Run the provided script with Python 3.11
python3.11 "$@"

