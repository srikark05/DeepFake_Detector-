# Fixes Applied for Model Training

## Issues Fixed

### 1. ✅ Missing Dependencies
- **Issue**: `scikit-learn` and other packages were not installed
- **Fix**: Installed all required packages except `giotto-tda`
- **Status**: ✅ Fixed

### 2. ✅ CMake Not Installed
- **Issue**: `giotto-tda` requires CMake to build
- **Fix**: Installed CMake via Homebrew
- **Status**: ✅ Fixed

### 3. ✅ PersistenceEntropy Implementation
- **Issue**: `PersistenceEntropy` from `gtda.diagrams` requires `giotto-tda >= 0.3.0`, but only 0.1.4 is available
- **Fix**: Created custom `PersistenceEntropy` implementation in `Homology and Topology/persistence_entropy.py`
- **Status**: ✅ Fixed - Code updated to use custom implementation

## Remaining Issue

### ❌ giotto-tda Installation Failure

**Problem**: The `giotto-tda` package (version 0.1.4) cannot be installed because:
1. Its build script tries to use `sudo` to install to system directories
2. This fails in non-interactive environments
3. The package appears to be outdated and not well-maintained

**Required Components from giotto-tda**:
- `CubicalPersistence` (used in `T_Diagrams.py`)
- `PersistenceLandscape` (used in `T_Diagrams.py`)
- `Amplitude` (used in `T_Diagrams.py`)
- `BettiCurve` (used in `T_Diagrams.py`)

## Solutions

### Option 1: Install Miniconda/Conda (Recommended)
```bash
# Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html
# Then:
conda install -c conda-forge giotto-tda
```

### Option 2: Use Python 3.10 or 3.11
The package may work better with older Python versions:
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Manual Build Fix
If you want to try fixing the build manually:
1. Download giotto-tda source
2. Modify the build script to not use sudo
3. Install manually

## Current Status

- ✅ All other dependencies installed
- ✅ Custom PersistenceEntropy implemented
- ❌ giotto-tda still needs to be installed via conda or alternative method

## Next Steps

1. Install conda/miniconda, OR
2. Use Python 3.10/3.11 in a virtual environment, OR
3. Contact the giotto-tda maintainers about the build issue

Once `giotto-tda` is installed, you should be able to run:
```bash
python3 train_model.py
```

