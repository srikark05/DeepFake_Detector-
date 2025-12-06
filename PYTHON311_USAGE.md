# Using Python 3.11 with giotto-tda

`giotto-tda` has been successfully installed with **Python 3.11**. Since your default Python is 3.13, you need to use Python 3.11 for scripts that require `giotto-tda`.

## Quick Start

### Option 1: Use the helper script
```bash
./use_python311.sh <script_name>.py [arguments]
```

### Option 2: Use python3.11 directly
```bash
python3.11 <script_name>.py [arguments]
```

## Examples

### Run persistence diagrams script:
```bash
cd "Results and Images"
python3.11 Persistence_Diagrams.py -m 5
```

### Run homology PCA analysis:
```bash
python3.11 Homology_PCA_Analysis.py -m 20
```

### Run the training pipeline:
```bash
python3.11 train_model.py
```

### Run the web app:
```bash
python3.11 app.py
```

## Installed Packages

The following packages are installed for Python 3.11:
- ✅ `giotto-tda` (version 0.6.2)
- ✅ All dependencies from `requirements.txt`
- ✅ FastAPI and web framework packages
- ✅ Matplotlib and visualization packages

## Verification

To verify giotto-tda is working:
```bash
python3.11 -c "from gtda.homology import CubicalPersistence; print('giotto-tda is working!')"
```

## Note

- Your default `python3` command uses Python 3.13 (which doesn't have giotto-tda)
- Use `python3.11` for scripts requiring giotto-tda
- All other scripts can continue using `python3` or `python3.11`

