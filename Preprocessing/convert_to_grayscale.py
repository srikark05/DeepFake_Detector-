#!/usr/bin/env python3
"""
Simple script to convert all images to grayscale
Run this from the project root directory
"""

import sys
import os
import shutil
import logging

# Check for required dependencies
try:
    from preprocess_images import ImagePreprocessor
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages with: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Convert all images in test/ and val/ folders to grayscale"""
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        input_dir="../preprocessing_raw_image_data",  # Raw image data directory
        output_dir="../preprocessing_raw_image_data",  # Output to same directory
        preserve_structure=False  # We'll create custom structure
    )
    
    logger.info("Starting grayscale conversion...")
    logger.info("This will convert all images and save them to test_gray/ and val_gray/ folders")
    
    # Process test dataset and save to test_gray
    logger.info("Processing test dataset...")
    test_processed, test_failed = preprocessor.process_dataset("test")
    
    # Rename test folder to test_gray
    test_gray_path = "../preprocessing_raw_image_data/test_gray"
    if os.path.exists("../preprocessing_raw_image_data/test"):
        if os.path.exists(test_gray_path):
            shutil.rmtree(test_gray_path)
        os.rename("../preprocessing_raw_image_data/test", test_gray_path)
        logger.info(f"Renamed test folder to test_gray")
    
    # Process validation dataset and save to val_gray
    logger.info("Processing validation dataset...")
    val_processed, val_failed = preprocessor.process_dataset("val")
    
    # Rename val folder to val_gray
    val_gray_path = "../preprocessing_raw_image_data/val_gray"
    if os.path.exists("../preprocessing_raw_image_data/val"):
        if os.path.exists(val_gray_path):
            shutil.rmtree(val_gray_path)
        os.rename("../preprocessing_raw_image_data/val", val_gray_path)
        logger.info(f"Renamed val folder to val_gray")
    
    # Combine results
    processed = test_processed + val_processed
    failed = test_failed + val_failed
    
    logger.info(f"Conversion complete!")
    logger.info(f"Successfully processed: {processed} images")
    logger.info(f"Failed to process: {failed} images")
    
    if failed > 0:
        logger.warning("Some images failed to process. Check the logs above for details.")

if __name__ == "__main__":
    main()
