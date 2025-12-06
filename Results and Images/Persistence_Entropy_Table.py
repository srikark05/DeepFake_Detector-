#!/usr/bin/env python3
"""
Persistence Entropy Table Generator
Creates tables showing persistence entropies for 50 images (25 real + 25 fake)
with all 4 filtrations (Height, Radial, Density, Dilation)
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocessing"))

try:
    from T_Diagrams import Persistence
    from pipeline import load_image, apply_filtrations_to_image
    from persistence_entropy import PersistenceEntropy
    GTDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Required modules not available: {e}")
    GTDA_AVAILABLE = False


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return image


def compute_entropy_for_image(image_path):
    """
    Compute persistence entropy for all 4 filtrations of an image
    
    Returns:
        dict: {filtration_name: entropy_value} where entropy_value is the sum of H0, H1, H2 entropies
    """
    if not GTDA_AVAILABLE:
        return None
    
    # Load image
    image = load_image(image_path)
    if image is None:
        return None
    
    image_norm = normalize_image(image)
    
    # Apply filtrations
    filtered_grids = apply_filtrations_to_image(image)
    
    # Initialize persistence module and entropy calculator
    persistence = Persistence(image_norm)
    entropy_calc = PersistenceEntropy()
    
    # Store entropies for each filtration
    entropies = {}
    filtration_names = ['height', 'radial', 'density', 'dilation']
    
    for filt_name in filtration_names:
        if filt_name not in filtered_grids:
            entropies[filt_name] = 0.0
            continue
        
        grid = filtered_grids[filt_name]
        grid_norm = normalize_image(grid)
        
        # Compute persistence diagram
        persistence.images = grid_norm
        diagrams = persistence.compute_diagram()
        
        # Compute entropy (returns [H0, H1, H2] entropies)
        entropy_values = entropy_calc.fit_transform(diagrams)
        
        # Sum entropies across all homology dimensions
        if entropy_values.ndim == 1:
            total_entropy = np.sum(entropy_values)
        else:
            total_entropy = np.sum(entropy_values.flatten())
        
        entropies[filt_name] = total_entropy
    
    return entropies, image


def create_entropy_table(images_data, label, output_path):
    """
    Create a table visualization with images on the left and entropy values on the right
    
    Args:
        images_data: List of tuples (image_path, image_array, entropies_dict, image_name)
        label: 'real' or 'fake'
        output_path: Path to save the table
    """
    n_images = len(images_data)
    filtration_names = ['height', 'radial', 'density', 'dilation']
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, max(20, n_images * 0.8)))
    gs = gridspec.GridSpec(n_images, 5, figure=fig, 
                           hspace=0.3, wspace=0.2,
                           width_ratios=[1, 1, 1, 1, 1])
    
    # Add title
    fig.suptitle(f'Persistence Entropy Table - {label.upper()} Images', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Column headers
    header_y = 0.98
    fig.text(0.1, header_y, 'Image', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.3, header_y, 'Height', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.5, header_y, 'Radial', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.7, header_y, 'Density', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.9, header_y, 'Dilation', fontsize=12, fontweight='bold', ha='center')
    
    # Process each image
    for row_idx, (img_path, img_array, entropies, img_name) in enumerate(images_data):
        # Image column
        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(img_array, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title(img_name[:20] + '...' if len(img_name) > 20 else img_name, 
                        fontsize=8, pad=2)
        
        # Entropy value columns
        for col_idx, filt_name in enumerate(filtration_names):
            ax_val = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax_val.axis('off')
            
            entropy_val = entropies.get(filt_name, 0.0)
            
            # Display value with appropriate formatting
            ax_val.text(0.5, 0.5, f'{entropy_val:.4f}', 
                       fontsize=10, ha='center', va='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {label} entropy table: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate persistence entropy tables for 50 images (25 real + 25 fake)'
    )
    parser.add_argument('--real-dir', type=str, 
                       default='../preprocessing_raw_image_data/val_gray/real',
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/fake',
                       help='Directory containing fake images')
    parser.add_argument('-o', '--output', type=str,
                       default='Persistence_Entropy_Tables',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=25,
                       help='Maximum number of images per class (default: 25)')
    
    args = parser.parse_args()
    
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required. Please install it first.")
        print("See INSTALL_GIOTTO.md for instructions.")
        sys.exit(1)
    
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image paths
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    
    if not real_dir.exists():
        print(f"Error: Real images directory not found: {real_dir}")
        sys.exit(1)
    if not fake_dir.exists():
        print(f"Error: Fake images directory not found: {fake_dir}")
        sys.exit(1)
    
    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    fake_images = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
    
    # Limit to max_images per class
    if args.max_images:
        real_images = real_images[:args.max_images]
        fake_images = fake_images[:args.max_images]
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    print(f"Processing {len(real_images) + len(fake_images)} images total...")
    print("="*60)
    
    # Process real images
    print("\nProcessing REAL images...")
    real_data = []
    for i, img_file in enumerate(real_images, 1):
        print(f"[{i}/{len(real_images)}] Processing: {img_file.name}")
        result = compute_entropy_for_image(img_file)
        if result is not None:
            entropies, image = result
            real_data.append((img_file, image, entropies, img_file.stem))
        else:
            print(f"  Warning: Failed to process {img_file.name}")
    
    # Process fake images
    print("\nProcessing FAKE images...")
    fake_data = []
    for i, img_file in enumerate(fake_images, 1):
        print(f"[{i}/{len(fake_images)}] Processing: {img_file.name}")
        result = compute_entropy_for_image(img_file)
        if result is not None:
            entropies, image = result
            fake_data.append((img_file, image, entropies, img_file.stem))
        else:
            print(f"  Warning: Failed to process {img_file.name}")
    
    # Create tables
    print("\n" + "="*60)
    print("Generating tables...")
    print("="*60)
    
    if len(real_data) > 0:
        real_output = output_dir / "Real_Images_Entropy_Table.png"
        create_entropy_table(real_data, 'real', real_output)
    
    if len(fake_data) > 0:
        fake_output = output_dir / "Fake_Images_Entropy_Table.png"
        create_entropy_table(fake_data, 'fake', fake_output)
    
    # Also create a combined table
    if len(real_data) > 0 and len(fake_data) > 0:
        all_data = real_data + fake_data
        combined_output = output_dir / "Combined_Entropy_Table.png"
        create_entropy_table(all_data, 'all', combined_output)
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Real images processed: {len(real_data)}")
    print(f"Fake images processed: {len(fake_data)}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

