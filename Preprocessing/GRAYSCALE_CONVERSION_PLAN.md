# Grayscale Conversion Plan for DeepFake Detection Dataset

## Overview
This document outlines the plan for converting all images in the `test/` and `val/` folders to grayscale format for the deepfake detection project.

## Dataset Structure
```
DeepFake_Detector-/
├── test/
│   ├── fake/     (1,606 files: 1,562 PNG + 44 JPG)
│   └── real/     (1,606 files: 1,577 PNG + 29 JPG)
├── val/
│   ├── fake/     (1,606 files: 1,468 PNG + 138 JPG)
│   └── real/     (1,606 files: 1,578 PNG + 28 JPG)
└── Total: 6,424 images
```

## Implementation

### 1. Main Preprocessing Script (`preprocess_images.py`)
- **Class-based design** for modularity and reusability
- **Batch processing** with progress bars using `tqdm`
- **Error handling** with detailed logging
- **Flexible output options** (overwrite or save to new directory)
- **Support for multiple image formats** (PNG, JPG, JPEG, BMP, TIFF)

### 2. Simple Usage Script (`convert_to_grayscale.py`)
- **One-command execution** for easy use
- **Automatic processing** of both test and validation datasets
- **Progress tracking** and error reporting

### 3. Requirements (`requirements.txt`)
- **Pillow**: Image processing library
- **NumPy**: Numerical operations
- **tqdm**: Progress bars

## Usage Instructions

### Option 1: Simple Conversion (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Convert all images to grayscale (overwrites original files)
python convert_to_grayscale.py
```

### Option 2: Advanced Usage
```bash
# Dry run to see what would be processed
python preprocess_images.py --dry_run

# Process only test dataset
python preprocess_images.py --dataset test

# Process only validation dataset
python preprocess_images.py --dataset val

# Save to new directory (preserve originals)
python preprocess_images.py --output_dir ./grayscale_data

# Process from specific input directory
python preprocess_images.py --input_dir /path/to/data --output_dir /path/to/output
```

## Key Features

### 1. **Preservation of Structure**
- Maintains the exact folder structure (`test/fake/`, `test/real/`, `val/fake/`, `val/real/`)
- Preserves original filenames

### 2. **Error Handling**
- Continues processing even if individual images fail
- Detailed logging of errors and success counts
- Graceful handling of corrupted or unsupported files

### 3. **Progress Tracking**
- Real-time progress bars for each dataset
- Clear indication of processing status
- Final summary of processed vs failed images

### 4. **Memory Efficient**
- Processes images one at a time
- Uses PIL's optimized image operations
- Automatic cleanup of image objects

## Expected Output

After running the conversion:
- All 6,424 images will be converted to grayscale
- Original folder structure will be preserved
- Images will be optimized for size (PIL's optimize=True)
- Processing time: ~5-10 minutes (depending on system)

## Quality Assurance

### Before Running:
1. **Backup your data** (recommended)
2. **Test with a small subset** first
3. **Check disk space** (grayscale images may be smaller)

### After Running:
1. **Verify file counts** match original
2. **Check a few sample images** to ensure proper conversion
3. **Review error logs** for any issues

## Troubleshooting

### Common Issues:
1. **Permission errors**: Ensure write access to directories
2. **Memory issues**: Process smaller batches if needed
3. **Corrupted images**: Check error logs for specific files

### Recovery:
- Original files are overwritten by default
- Use `--output_dir` to preserve originals
- Re-run with `--dataset` to process specific folders

## Next Steps

After grayscale conversion:
1. **Verify conversion quality**
2. **Proceed with model training**
3. **Implement TDA pipeline** on grayscale data
4. **Create data loaders** for training

## File Structure After Conversion
```
DeepFake_Detector-/
├── test/
│   ├── fake/     (1,606 grayscale images)
│   └── real/     (1,606 grayscale images)
├── val/
│   ├── fake/     (1,606 grayscale images)
│   └── real/     (1,606 grayscale images)
├── preprocess_images.py
├── convert_to_grayscale.py
├── requirements.txt
└── GRAYSCALE_CONVERSION_PLAN.md
```
