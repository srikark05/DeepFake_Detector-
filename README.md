# DeepFake_Detector-
This is a repo for a DeepFake classification model that utilizes a TDA pipeline in detection.

## Project Structure

- **`preprocessing_raw_image_data/`** - Image datasets
  - `test_gray/` - Grayscale test dataset (fake + real images)
  - `val_gray/` - Grayscale validation dataset (fake + real images)
- **`preprocessing/`** - Image preprocessing tools and documentation
- **`Filtration/`** - Filtration module (height, radial, density, dilation)
- **`Homology and Topology/`** - Persistence diagrams and feature extraction
- **`ML Models/`** - SVM classifier implementation
- **`pipeline.py`** - Complete end-to-end pipeline
- **`docs/`** - Project proposal and academic literature

## Quick Start

### 1. Preprocess Images to Grayscale

```bash
cd preprocessing
pip install -r requirements.txt
python convert_to_grayscale.py
```

### 2. Install Pipeline Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline

```bash
python pipeline.py
```

This will:
- Load grayscale images
- Extract TDA features using filtrations and persistence diagrams
- Train an SVM classifier
- Display comprehensive results (accuracy, confusion matrix, classification report)

### 4. Test Components (Optional)

```bash
python test_pipeline.py
```

## Pipeline Overview

The pipeline integrates:
1. **Data Loading**: Loads images from `preprocessing_raw_image_data/`
2. **Filtration**: Applies 4 types of filtrations (height, radial, density, dilation)
3. **Persistence Diagrams**: Computes cubical persistence
4. **Feature Extraction**: Extracts TDA features (landscapes, entropy, amplitudes, Betti curves)
5. **Classification**: Trains SVM and evaluates performance

See `PIPELINE_README.md` for detailed documentation.

## Web Application

A modern web interface is available for easy image classification:

### Quick Start

1. **Train the model** (if not already done):
   ```bash
   python train_model.py
   ```

2. **Start the web server**:
   ```bash
   python app.py
   ```
   
   Or use the convenience script:
   ```bash
   ./start_server.sh
   ```

3. **Open in browser**: Navigate to `http://localhost:8000`

### Features

- 🎨 Beautiful drag-and-drop interface
- 📊 Real-time predictions with confidence scores
- 🖼️ Image preview
- 📱 Responsive design

See `WEB_APP_README.md` for detailed web app documentation. 
