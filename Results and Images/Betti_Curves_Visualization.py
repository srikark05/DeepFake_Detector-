#!/usr/bin/env python3
"""
Betti Curves Visualization Generator
Generates Betti curves for a random subsample of 40 images (20 real + 20 fake)
showing all 4 filtrations (Height, Radial, Density, Dilation)
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocessing"))

try:
    from T_Diagrams import Persistence
    from pipeline import load_image, apply_filtrations_to_image
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


def compute_betti_curves_for_image(image_path):
    """
    Compute Betti curves for all 4 filtrations of an image
    
    Returns:
        dict: {filtration_name: betti_curve_array}
        image: The loaded image array
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
    
    # Initialize persistence module
    persistence = Persistence(image_norm)
    
    # Store Betti curves for each filtration
    betti_curves = {}
    filtration_names = ['height', 'radial', 'density', 'dilation']
    
    for filt_name in filtration_names:
        if filt_name not in filtered_grids:
            betti_curves[filt_name] = None
            continue
        
        grid = filtered_grids[filt_name]
        grid_norm = normalize_image(grid)
        
        # Compute persistence diagram
        persistence.images = grid_norm
        diagrams = persistence.compute_diagram()
        
        # Compute Betti curves
        betti = persistence.compute_betti(diagrams)
        betti_curves[filt_name] = betti[0]  # Single image
    
    return betti_curves, image


def plot_betti_curve(betti_curve, title="Betti Curve", ax=None):
    """
    Plot Betti curve showing H0, H1, H2 over filtration parameter
    
    Args:
        betti_curve: Array from compute_betti, shape (n_samples, n_homology_dims)
        title: Plot title
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    if betti_curve is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return ax
    
    # Reshape if needed
    if betti_curve.ndim == 1:
        # Assume it's flattened, try to reshape
        n_dims = 3  # H0, H1, H2
        n_samples = len(betti_curve) // n_dims
        if n_samples > 0:
            betti_reshaped = betti_curve.reshape(n_samples, n_dims)
        else:
            betti_reshaped = betti_curve.reshape(1, -1)
    else:
        betti_reshaped = betti_curve
    
    # Extract curves for each dimension
    x = np.arange(betti_reshaped.shape[0])
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}
    
    for dim in range(min(betti_reshaped.shape[1], 3)):
        if betti_reshaped.shape[1] > dim:
            ax.plot(x, betti_reshaped[:, dim], 
                   color=colors[dim], label=labels[dim], 
                   linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Filtration Parameter', fontsize=8)
    ax.set_ylabel('Betti Number', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def create_single_image_betti_visualization(img_path, img_array, betti_curves, img_name, label, output_dir):
    """
    Create a visualization of Betti curves for a single image with all 4 filtrations
    
    Args:
        img_path: Path to the image
        img_array: Image array
        betti_curves: Dictionary of betti curves for each filtration
        img_name: Name of the image
        label: 'real' or 'fake'
        output_dir: Directory to save the visualization
    """
    filtration_names = ['height', 'radial', 'density', 'dilation']
    
    # Create figure with 2 rows: image + 4 Betti curves
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, 
                           hspace=0.4, wspace=0.3,
                           width_ratios=[1, 1, 1, 1, 1])
    
    # Row 1: Original image + 4 Betti curves
    # Original image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_array, cmap='gray')
    ax_img.set_title('Original Image', fontsize=10, fontweight='bold')
    ax_img.axis('off')
    
    # Betti curves for each filtration
    for idx, filt_name in enumerate(filtration_names):
        ax_betti = fig.add_subplot(gs[0, idx + 1])
        betti_curve = betti_curves.get(filt_name)
        plot_betti_curve(betti_curve, 
                       title=f'{filt_name.capitalize()} Filtration', 
                       ax=ax_betti)
    
    # Row 2: Empty space + 4 Betti curves (duplicate for better visibility)
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')
    
    for idx, filt_name in enumerate(filtration_names):
        ax_betti = fig.add_subplot(gs[1, idx + 1])
        betti_curve = betti_curves.get(filt_name)
        plot_betti_curve(betti_curve, 
                       title=f'{filt_name.capitalize()} (Detail)', 
                       ax=ax_betti)
    
    # Add overall title
    plt.suptitle(f'Betti Curves: {label.upper()} - {img_name}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{label}_{img_name}_betti_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Betti curves for random subsample of 40 images with all 4 filtrations'
    )
    parser.add_argument('--real-dir', type=str, 
                       default='../preprocessing_raw_image_data/val_gray/real',
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/fake',
                       help='Directory containing fake images')
    parser.add_argument('-o', '--output', type=str,
                       default='Betti_Curves_Results',
                       help='Output directory')
    parser.add_argument('-n', '--num-samples', type=int, default=40,
                       help='Number of images to sample (default: 40)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required. Please install it first.")
        print("See INSTALL_GIOTTO.md for instructions.")
        sys.exit(1)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
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
    
    all_real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    all_fake_images = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
    
    # Randomly sample images (half real, half fake)
    n_per_class = args.num_samples // 2
    n_per_class = min(n_per_class, len(all_real_images), len(all_fake_images))
    
    real_images = random.sample(all_real_images, n_per_class)
    fake_images = random.sample(all_fake_images, n_per_class)
    
    print(f"Randomly sampled {len(real_images)} real images and {len(fake_images)} fake images")
    print(f"Total: {len(real_images) + len(fake_images)} images")
    print(f"Processing with seed={args.seed}...")
    print("="*60)
    
    # Process all images
    all_images_data = []
    
    # Process real images
    print("\nProcessing REAL images...")
    for i, img_file in enumerate(real_images, 1):
        print(f"[{i}/{len(real_images)}] Processing: {img_file.name}")
        result = compute_betti_curves_for_image(img_file)
        if result is not None:
            betti_curves, image = result
            all_images_data.append((img_file, image, betti_curves, img_file.stem, 'real'))
        else:
            print(f"  Warning: Failed to process {img_file.name}")
    
    # Process fake images
    print("\nProcessing FAKE images...")
    for i, img_file in enumerate(fake_images, 1):
        print(f"[{i}/{len(fake_images)}] Processing: {img_file.name}")
        result = compute_betti_curves_for_image(img_file)
        if result is not None:
            betti_curves, image = result
            all_images_data.append((img_file, image, betti_curves, img_file.stem, 'fake'))
        else:
            print(f"  Warning: Failed to process {img_file.name}")
    
    # Create individual visualizations for each image
    if len(all_images_data) > 0:
        print("\n" + "="*60)
        print("Generating individual Betti curves visualizations...")
        print("="*60)
        
        for img_path, img_array, betti_curves, img_name, label in all_images_data:
            create_single_image_betti_visualization(
                img_path, img_array, betti_curves, img_name, label, output_dir
            )
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Total images processed: {len(all_images_data)}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

