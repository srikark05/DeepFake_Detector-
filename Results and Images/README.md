# Results and Images

This folder contains scripts and results for generating and visualizing TDA analysis including filtration diagrams, persistence diagrams, barcodes, landscapes, and PCA analysis.

## Structure

```
Results and Images/
├── README.md                          # This file
├── Filtration_Results.py              # Script to generate filtration diagrams
├── Persistence_Diagrams.py            # Script to generate persistence diagrams, barcodes, landscapes
├── Homology_PCA_Analysis.py           # Script for homology analysis and 2D PCA visualization
├── Filtration_Results/                # Output folder for filtration diagrams
├── Persistence_Results/               # Output folder for persistence analysis
│   ├── real/                          # Real image persistence results
│   └── fake/                          # Fake image persistence results
└── Homology_PCA_Results/              # Output folder for PCA analysis
    ├── 2D_PCA_Persistence_Landscapes.png
    ├── 2D_PCA_Homology_Features.png
    └── Homology_Features_Comparison.png
```

## Filtration_Results.py

This script applies four types of filtrations to images and generates visualization diagrams:

1. **Height Filtration**: Projects points onto a direction vector (intensity-based)
2. **Radial Filtration**: Computes distance from center point
3. **Density Filtration**: Calculates local point density using k-nearest neighbors
4. **Dilation Filtration**: Computes pairwise distances between points

### Usage

#### Process a single image:
```bash
python Filtration_Results.py <path_to_image>
```

#### Process multiple images from a directory:
```bash
python Filtration_Results.py <path_to_directory> -m 10
```

#### Specify custom output directory:
```bash
python Filtration_Results.py <path_to_image> -o custom_output_dir
```

#### Adjust subsampling (for performance):
```bash
python Filtration_Results.py <path_to_image> -s 10000
```

### Command-line Options

- `input`: Input image file or directory (required)
- `-o, --output`: Output directory (default: `Filtration_Results`)
- `-s, --subsample`: Number of points to subsample (default: 5000)
- `-m, --max-images`: Maximum number of images to process (for directories)

### Examples

```bash
# Process a single image
python Filtration_Results.py ../preprocessing_raw_image_data/val_gray/real/00000026_29.png

# Process first 5 images from a directory
python Filtration_Results.py ../preprocessing_raw_image_data/val_gray/real -m 5

# Process with higher subsampling for better quality
python Filtration_Results.py ../preprocessing_raw_image_data/test_gray/fake/00000001.png -s 10000
```

### Output

For each processed image, the script generates:

1. **Combined Diagram** (`<image_name>_filtration_diagram.png`):
   - Shows original image and all 4 filtrations in a grid layout
   - Includes colorbars for each filtration type

2. **Individual Filtration Images**:
   - `<image_name>_height_filtration.png` - Height filtration visualization
   - `<image_name>_radial_filtration.png` - Radial filtration visualization
   - `<image_name>_density_filtration.png` - Density filtration visualization
   - `<image_name>_dilation_filtration.png` - Dilation filtration visualization

### Color Maps

Each filtration type uses a different colormap for visualization:
- **Height**: Viridis (green-blue)
- **Radial**: Plasma (purple-yellow)
- **Density**: Inferno (black-red-yellow)
- **Dilation**: Magma (black-purple-pink)

### Performance Notes

- Default subsampling is 5000 points for performance
- Increase `-s` value for higher quality (slower processing)
- Processing time depends on image size and subsampling level
- Large batches may take significant time

### Dependencies

- `numpy` - Numerical operations
- `scipy` - Spatial operations (KDTree, distance calculations)
- `PIL/Pillow` - Image loading
- `matplotlib` - Visualization

All dependencies are included in the main `requirements.txt`.

---

## Persistence_Diagrams.py

This script generates persistence diagrams (H0, H1, H2), barcodes, and persistence landscapes for real and fake images.

### Features

- **Persistence Diagrams**: Visualizes H0, H1, and H2 homology groups
- **Barcodes**: Shows persistence intervals for each homology dimension
- **Persistence Landscapes**: Displays landscape functions for topological features
- **Separate Visualizations**: Individual diagrams for each homology dimension

### Usage

```bash
# Process default directories (5 images per class)
python Persistence_Diagrams.py

# Custom directories and number of images
python Persistence_Diagrams.py --real-dir ../preprocessing_raw_image_data/val_gray/real \
                               --fake-dir ../preprocessing_raw_image_data/val_gray/fake \
                               -m 10

# Custom output directory
python Persistence_Diagrams.py -o Custom_Persistence_Results
```

### Command-line Options

- `--real-dir`: Directory containing real images (default: `../preprocessing_raw_image_data/val_gray/real`)
- `--fake-dir`: Directory containing fake images (default: `../preprocessing_raw_image_data/val_gray/fake`)
- `-o, --output`: Output directory (default: `Persistence_Results`)
- `-m, --max-images`: Maximum number of images per class (default: 5)

### Output

For each processed image, generates:

1. **Combined Analysis** (`<label>_<image_name>_persistence_analysis.png`):
   - Original image
   - Persistence diagram (H0, H1, H2)
   - Persistence barcode
   - Persistence landscape

2. **Individual H0, H1, H2 Diagrams**:
   - `<label>_<image_name>_H0_diagram.png`
   - `<label>_<image_name>_H1_diagram.png`
   - `<label>_<image_name>_H2_diagram.png`

3. **Barcode** (`<label>_<image_name>_barcode.png`)

4. **Landscape** (`<label>_<image_name>_landscape.png`)

### Requirements

- `giotto-tda` must be installed (see `INSTALL_GIOTTO.md`)

---

## Homology_PCA_Analysis.py

This script performs homology analysis and 2D Principal Component Analysis (PCA) on persistence landscapes to detect linear separability between real and fake images.

### Features

- **Persistence Landscape Extraction**: Extracts landscapes from images
- **2D PCA Visualization**: Projects landscapes to 2D space for visualization
- **Homology Feature Analysis**: Computes statistics for H0, H1, H2
- **Separability Analysis**: Calculates silhouette scores to measure class separation
- **Comparative Visualization**: Compares homology features between real and fake images

### Usage

```bash
# Process default directories (20 images per class)
python Homology_PCA_Analysis.py

# Custom directories and number of images
python Homology_PCA_Analysis.py --real-dir ../preprocessing_raw_image_data/val_gray/real \
                                 --fake-dir ../preprocessing_raw_image_data/val_gray/fake \
                                 -m 30

# Custom output directory
python Homology_PCA_Analysis.py -o Custom_PCA_Results
```

### Command-line Options

- `--real-dir`: Directory containing real images (default: `../preprocessing_raw_image_data/val_gray/real`)
- `--fake-dir`: Directory containing fake images (default: `../preprocessing_raw_image_data/val_gray/fake`)
- `-o, --output`: Output directory (default: `Homology_PCA_Results`)
- `-m, --max-images`: Maximum number of images per class (default: 20)

### Output

1. **2D PCA of Persistence Landscapes** (`2D_PCA_Persistence_Landscapes.png`):
   - Scatter plot showing real (green) and fake (red) images in 2D PCA space
   - Shows explained variance for each principal component
   - Includes silhouette score for separability assessment

2. **2D PCA of Homology Features** (`2D_PCA_Homology_Features.png`):
   - PCA visualization using computed homology statistics
   - Shows separability based on feature vectors

3. **Homology Features Comparison** (`Homology_Features_Comparison.png`):
   - Bar charts comparing H0, H1, H2 features between real and fake
   - Shows count, mean persistence, max persistence, total persistence
   - Summary statistics table

### Analysis Metrics

- **Explained Variance**: Percentage of variance captured by each principal component
- **Silhouette Score**: Measures how well-separated the classes are
  - > 0.5: Good separation
  - 0.3-0.5: Moderate separation
  - < 0.3: Poor separation

### Homology Features

For each image, computes:
- **H0 Features**: Connected components statistics
- **H1 Features**: Loop/hole statistics
- **H2 Features**: Void/cavity statistics

Each dimension includes:
- Count of features
- Mean persistence
- Maximum persistence
- Total persistence

### Requirements

- `giotto-tda` must be installed (see `INSTALL_GIOTTO.md`)
- `scikit-learn` for PCA and metrics

---

## 3D_PCA_Analysis.py

This script performs **3D Principal Component Analysis** on persistence landscapes and homology features, providing a three-dimensional visualization of the feature space to better understand separability between real and fake images.

### Features

- **3D PCA Visualization**: Projects features to 3D space (PC1, PC2, PC3) for better visualization
- **Persistence Landscape Analysis**: 3D PCA on persistence landscapes
- **Homology Feature Analysis**: 3D PCA on computed homology statistics
- **Multiple Viewing Angles**: Optional generation of multiple 3D views from different angles
- **Separability Metrics**: Calculates silhouette scores in 3D space

### Usage

```bash
# Process 100 images per class (default)
python3.11 3D_PCA_Analysis.py -m 100

# Custom number of images
python3.11 3D_PCA_Analysis.py -m 50

# Generate multiple viewing angles
python3.11 3D_PCA_Analysis.py -m 100 --views

# Custom directories
python3.11 3D_PCA_Analysis.py --real-dir ../preprocessing_raw_image_data/val_gray/real \
                               --fake-dir ../preprocessing_raw_image_data/val_gray/fake \
                               -m 100
```

### Command-line Options

- `--real-dir`: Directory containing real images (default: `../preprocessing_raw_image_data/val_gray/real`)
- `--fake-dir`: Directory containing fake images (default: `../preprocessing_raw_image_data/val_gray/fake`)
- `-o, --output`: Output directory (default: `Homology_PCA_Results`)
- `-m, --max-images`: Maximum number of images per class (default: 100)
- `--views`: Generate multiple 3D viewing angles (standard, side, top, front, back_side)

### Output

1. **3D PCA of Persistence Landscapes** (`3D_PCA_Persistence_Landscapes.png`):
   - 3D scatter plot showing real (green) and fake (red) images
   - Shows explained variance for PC1, PC2, PC3
   - Includes silhouette score for separability assessment
   - Interactive 3D visualization (can be rotated in image viewer)

2. **3D PCA of Homology Features** (`3D_PCA_Homology_Features.png`):
   - 3D PCA visualization using computed homology statistics
   - Shows separability in 3D feature space

3. **Multiple Views** (if `--views` flag is used):
   - `3D_PCA_Persistence_Landscapes_standard_view.png`
   - `3D_PCA_Persistence_Landscapes_side_view.png`
   - `3D_PCA_Persistence_Landscapes_top_view.png`
   - `3D_PCA_Persistence_Landscapes_front_view.png`
   - `3D_PCA_Persistence_Landscapes_back_side_view.png`
   - Same for homology features

### Advantages of 3D PCA

- **Better Visualization**: 3D space can reveal separability not visible in 2D
- **More Variance Captured**: Includes PC3, capturing more information
- **Spatial Understanding**: Better sense of data distribution in feature space
- **Rotation**: Can view from different angles to find best separation view

### Analysis Metrics

- **Explained Variance**: Percentage captured by PC1, PC2, PC3
- **Total Variance**: Sum of all three components
- **Silhouette Score**: Measures class separation in 3D space
  - > 0.5: Good separation
  - 0.3-0.5: Moderate separation
  - < 0.3: Poor separation

### Requirements

- `giotto-tda` must be installed (see `INSTALL_GIOTTO.md`)
- `scikit-learn` for PCA and metrics
- `matplotlib` with 3D support (`mpl_toolkits.mplot3d`)

---

## Quick Start Guide

### 1. Generate Filtration Diagrams
```bash
cd "Results and Images"
python Filtration_Results.py ../preprocessing_raw_image_data/val_gray/real/00000026_29.png
```

### 2. Generate Persistence Diagrams
```bash
python Persistence_Diagrams.py -m 5
```

### 3. Perform 2D Homology PCA Analysis
```bash
python3.11 Homology_PCA_Analysis.py -m 20
```

### 4. Perform 3D Homology PCA Analysis
```bash
python3.11 3D_PCA_Analysis.py -m 100
```

### Note on giotto-tda

All persistence-related scripts require `giotto-tda` to be installed. If you encounter import errors:

1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Install giotto-tda: `conda install -c conda-forge giotto-tda`
3. Or use Python 3.10/3.11 instead of 3.13

See `../INSTALL_GIOTTO.md` for detailed instructions.

