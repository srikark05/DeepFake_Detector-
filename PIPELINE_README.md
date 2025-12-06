# TDA-Based DeepFake Detection Pipeline

This document describes the complete pipeline for deepfake detection using Topological Data Analysis (TDA).

## Overview

The pipeline integrates the following components:
1. **Data Loading**: Loads grayscale images from `preprocessing_raw_image_data/`
2. **Filtration**: Applies height, radial, density, and dilation filtrations
3. **Persistence Diagrams**: Computes cubical persistence diagrams
4. **Feature Extraction**: Extracts TDA features (landscapes, entropy, amplitudes, Betti curves)
5. **Classification**: Trains an SVM classifier and evaluates performance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning
- `giotto-tda` - Topological Data Analysis
- `tqdm` - Progress bars

## Usage

### Running the Complete Pipeline

```bash
python pipeline.py
```

The pipeline will:
1. Load images from `preprocessing_raw_image_data/test_gray/` and `preprocessing_raw_image_data/val_gray/`
2. Extract TDA features from all images
3. Train an SVM classifier
4. Display comprehensive results including:
   - Accuracy
   - Classification report
   - Confusion matrix
   - Precision, Recall, F1-Score

### Testing Components

Before running the full pipeline, test that all components work:

```bash
python test_pipeline.py
```

This will verify:
- All modules can be imported
- Filtration class works correctly
- Persistence diagrams can be computed
- Data can be loaded

## Pipeline Structure

```
pipeline.py
├── Data Loading
│   └── load_dataset() - Loads images from test_gray and val_gray
├── Image Processing
│   └── image_to_point_cloud() - Converts images to 3D point clouds
├── Filtration
│   └── apply_filtrations_to_image() - Applies 4 types of filtrations
├── TDA Feature Extraction
│   └── extract_tda_features_for_image() - Extracts comprehensive TDA features
├── Classification
│   └── TDASVMClassifier - Trains and evaluates SVM model
└── Results
    └── Comprehensive evaluation metrics
```

## Data Structure

The pipeline expects the following structure:

```
preprocessing_raw_image_data/
├── test_gray/
│   ├── fake/     (grayscale images)
│   └── real/     (grayscale images)
└── val_gray/
    ├── fake/     (grayscale images)
    └── real/     (grayscale images)
```

## Features Extracted

For each image, the pipeline extracts:

1. **Height Filtration Features**: Based on intensity values
2. **Radial Filtration Features**: Distance from image center
3. **Density Filtration Features**: Local density of points
4. **Dilation Filtration Features**: Distance-based features

Each filtration produces:
- Persistence landscapes
- Persistence entropy
- Wasserstein amplitudes
- Betti curves

## Output

The pipeline provides:

1. **Accuracy**: Overall classification accuracy
2. **Classification Report**: Precision, recall, F1-score per class
3. **Confusion Matrix**: True/False positives and negatives
4. **Additional Metrics**: Precision, Recall, F1-Score

## Performance Notes

- Images are resized to 128x128 for efficiency
- Point clouds are subsampled to 5000 points for filtration
- For full dataset, remove `max_images_per_class` limit in `load_dataset()`

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that module paths are correct

### Data Loading Errors
- Verify `preprocessing_raw_image_data/` directory exists
- Ensure `test_gray/` and `val_gray/` folders contain images

### Memory Issues
- Reduce `max_images_per_class` parameter
- Reduce image size in `target_size`
- Reduce point cloud subsample size

## Next Steps

After running the pipeline:
1. Review the results and metrics
2. Adjust hyperparameters if needed (C, gamma in SVM)
3. Experiment with different feature combinations
4. Try different classifiers or ensemble methods


