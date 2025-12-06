#!/usr/bin/env python3
"""
Image Preprocessing Script for DeepFake Detection
Converts images in test/ and val/ folders to grayscale
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Check for required dependencies
try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install required packages with: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing operations for deepfake detection"""
    
    def __init__(self, input_dir, output_dir=None, preserve_structure=True):
        """
        Initialize the preprocessor
        
        Args:
            input_dir (str): Directory containing test/ and val/ folders
            output_dir (str): Output directory (if None, overwrites original)
            preserve_structure (bool): Whether to preserve folder structure
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.preserve_structure = preserve_structure
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
    def convert_to_grayscale(self, image_path, output_path):
        """
        Convert a single image to grayscale
        
        Args:
            image_path (Path): Path to input image
            output_path (Path): Path to save grayscale image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != 'L':
                    grayscale_img = img.convert('L')
                else:
                    grayscale_img = img.copy()
                
                # Create output directory if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save grayscale image
                grayscale_img.save(output_path, optimize=True)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def get_image_files(self, directory):
        """
        Get all image files from a directory recursively
        
        Args:
            directory (Path): Directory to search
            
        Returns:
            list: List of image file paths
        """
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
        return sorted(image_files)
    
    def process_dataset(self, dataset_name="both"):
        """
        Process images in test/ and val/ folders
        
        Args:
            dataset_name (str): "test", "val", or "both"
        """
        datasets = []
        if dataset_name in ["test", "both"]:
            datasets.append("test")
        if dataset_name in ["val", "both"]:
            datasets.append("val")
        
        total_processed = 0
        total_failed = 0
        
        for dataset in datasets:
            dataset_path = self.input_dir / dataset
            if not dataset_path.exists():
                logger.warning(f"Dataset folder {dataset_path} does not exist, skipping...")
                continue
                
            logger.info(f"Processing {dataset} dataset...")
            
            # Process fake images
            fake_path = dataset_path / "fake"
            if fake_path.exists():
                fake_files = self.get_image_files(fake_path)
                logger.info(f"Found {len(fake_files)} fake images in {dataset}")
                
                for img_path in tqdm(fake_files, desc=f"Processing {dataset}/fake"):
                    if self.preserve_structure:
                        rel_path = img_path.relative_to(self.input_dir)
                        output_path = self.output_dir / rel_path
                    else:
                        output_path = self.output_dir / dataset / "fake" / img_path.name
                    
                    if self.convert_to_grayscale(img_path, output_path):
                        total_processed += 1
                    else:
                        total_failed += 1
            
            # Process real images
            real_path = dataset_path / "real"
            if real_path.exists():
                real_files = self.get_image_files(real_path)
                logger.info(f"Found {len(real_files)} real images in {dataset}")
                
                for img_path in tqdm(real_files, desc=f"Processing {dataset}/real"):
                    if self.preserve_structure:
                        rel_path = img_path.relative_to(self.input_dir)
                        output_path = self.output_dir / rel_path
                    else:
                        output_path = self.output_dir / dataset / "real" / img_path.name
                    
                    if self.convert_to_grayscale(img_path, output_path):
                        total_processed += 1
                    else:
                        total_failed += 1
        
        logger.info(f"Processing complete! Processed: {total_processed}, Failed: {total_failed}")
        return total_processed, total_failed

def main():
    """Main function to run the preprocessing"""
    parser = argparse.ArgumentParser(description="Convert images to grayscale for deepfake detection")
    parser.add_argument("--input_dir", default=".", help="Input directory containing test/ and val/ folders")
    parser.add_argument("--output_dir", default=None, help="Output directory (if None, overwrites original)")
    parser.add_argument("--dataset", choices=["test", "val", "both"], default="both", 
                       help="Which dataset to process")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually doing it")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preserve_structure=True
    )
    
    if args.dry_run:
        logger.info("DRY RUN - Showing what would be processed:")
        for dataset in ["test", "val"]:
            dataset_path = Path(args.input_dir) / dataset
            if dataset_path.exists():
                for subfolder in ["fake", "real"]:
                    subfolder_path = dataset_path / subfolder
                    if subfolder_path.exists():
                        files = preprocessor.get_image_files(subfolder_path)
                        logger.info(f"{dataset}/{subfolder}: {len(files)} files")
    else:
        # Process the dataset
        preprocessor.process_dataset(args.dataset)

if __name__ == "__main__":
    main()
