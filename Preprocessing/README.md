# Image Preprocessing for DeepFake Detection

This folder contains all the tools needed to convert images to grayscale for the deepfake detection project.

## Files

- **`convert_to_grayscale.py`** - Simple one-command script to convert all images
- **`preprocess_images.py`** - Advanced preprocessing class with more options
- **`requirements.txt`** - Python dependencies needed
- **`GRAYSCALE_CONVERSION_PLAN.md`** - Detailed documentation and usage guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Convert all images to grayscale:**
   ```bash
   python convert_to_grayscale.py
   ```

## Usage from Project Root

If running from the main project directory, use:
```bash
cd preprocessing
python convert_to_grayscale.py
```

## Dataset Structure

The scripts expect the following structure in the parent directory:
```
DeepFake_Detector-/
├── preprocessing_raw_image_data/
│   ├── test/                    (Original color images)
│   │   ├── fake/     (1,606 images)
│   │   └── real/     (1,606 images)
│   ├── val/                     (Original color images)
│   │   ├── fake/     (1,606 images)
│   │   └── real/     (1,606 images)
│   ├── test_gray/               (Generated grayscale images)
│   │   ├── fake/     (1,606 images)
│   │   └── real/     (1,606 images)
│   └── val_gray/                (Generated grayscale images)
│       ├── fake/     (1,606 images)
│       └── real/     (1,606 images)
└── preprocessing/
    ├── convert_to_grayscale.py
    ├── preprocess_images.py
    ├── requirements.txt
    └── README.md
```

## Features

- ✅ Converts all 6,424 images to grayscale
- ✅ Creates separate `test_gray/` and `val_gray/` folders
- ✅ Preserves original images in `test/` and `val/` folders
- ✅ Progress tracking with visual bars
- ✅ Error handling and logging
- ✅ Flexible output options
- ✅ Support for multiple image formats

## Next Steps

After grayscale conversion, your images will be ready for:
1. Model training
2. TDA pipeline implementation
3. Feature extraction
