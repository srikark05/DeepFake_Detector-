# Installing giotto-tda

The `giotto-tda` package is required for the TDA pipeline but can be difficult to install. Here are several options:

## Option 1: Install via Conda (Recommended)

If you have conda/miniconda installed:

```bash
conda install -c conda-forge giotto-tda
```

This is usually the most reliable method.

## Option 2: Install with User Flag

Try installing with the `--user` flag to avoid system directory permissions:

```bash
pip install --user giotto-tda==0.1.4
```

## Option 3: Use Python 3.10 or 3.11

`giotto-tda` may have better compatibility with Python 3.10 or 3.11. Consider using a virtual environment:

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

## Option 4: Install Build Dependencies First

Make sure you have all build dependencies:

```bash
# On macOS
brew install cmake
brew install boost
brew install eigen

# Then try installing
pip install giotto-tda==0.1.4
```

## Note

The code has been updated to use a custom `PersistenceEntropy` implementation, so that specific class no longer requires `giotto-tda`. However, you still need `giotto-tda` for:
- `CubicalPersistence`
- `PersistenceLandscape`
- `Amplitude`
- `BettiCurve`

These are used in `Homology and Topology/T_Diagrams.py`.

## Alternative: Minimal Installation

If installation continues to fail, you may need to:
1. Use a different Python version (3.10 or 3.11)
2. Use conda instead of pip
3. Install from source with proper build environment

