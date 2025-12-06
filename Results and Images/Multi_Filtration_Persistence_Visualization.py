#!/usr/bin/env python3
"""
Multi-Filtration Persistence Diagrams and Barcodes Generator
Generates persistence diagrams and barcodes for 40 images (20 real + 20 fake)
showing all 4 filtrations (Height, Radial, Density, Dilation) for each image
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocessing"))

try:
    from T_Diagrams import Persistence
    from pipeline import load_image, image_to_point_cloud, apply_filtrations_to_image
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


def separate_diagrams_by_dimension(diagram):
    """
    Separate persistence diagram by homology dimension
    
    Args:
        diagram: Array of shape (n_points, 3) with [dim, birth, death]
    
    Returns:
        dict: {0: h0_points, 1: h1_points, 2: h2_points}
    """
    separated = {0: [], 1: [], 2: []}
    
    for point in diagram:
        dim = int(point[0])
        if dim in separated:
            separated[dim].append([point[1], point[2]])  # [birth, death]
    
    # Convert to numpy arrays
    for dim in separated:
        if len(separated[dim]) > 0:
            separated[dim] = np.array(separated[dim])
        else:
            separated[dim] = np.array([]).reshape(0, 2)
    
    return separated


def plot_persistence_diagram(diagram, title="Persistence Diagram", ax=None):
    """
    Plot persistence diagram with different colors for each dimension
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    separated = separate_diagrams_by_dimension(diagram)
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}
    
    for dim in [0, 1, 2]:
        points = separated[dim]
        if len(points) > 0:
            births = points[:, 0]
            deaths = points[:, 1]
            ax.scatter(births, deaths, c=colors[dim], label=labels[dim], 
                      alpha=0.6, s=20)
    
    # Plot diagonal line
    if len(diagram) > 0:
        max_val = max(diagram[:, 1:].max(), 1.0)
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Birth', fontsize=8)
    ax.set_ylabel('Death', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    if len(separated[0]) > 0 or len(separated[1]) > 0 or len(separated[2]) > 0:
        ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_barcode(diagram, title="Persistence Barcode", ax=None):
    """
    Plot persistence barcode for each homology dimension
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    separated = separate_diagrams_by_dimension(diagram)
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}
    
    y_pos = 0
    y_ticks = []
    y_labels = []
    
    for dim in [0, 1, 2]:
        points = separated[dim]
        if len(points) > 0:
            valid_points = points[points[:, 1] > points[:, 0]]  # Only persistent features
            if len(valid_points) > 0:
                start_y = y_pos
                for birth, death in valid_points:
                    ax.plot([birth, death], [y_pos, y_pos], 
                           color=colors[dim], linewidth=1.5, alpha=0.7)
                    y_pos += 1
                
                if y_pos > start_y:
                    y_ticks.append((start_y + y_pos - 1) / 2)
                    y_labels.append(labels[dim])
                    y_pos += 1  # Space between dimensions
    
    if len(y_ticks) > 0:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=6)
    
    ax.set_xlabel('Filtration Value', fontsize=8)
    ax.set_ylabel('Feature', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_persistence_landscape(landscape, title="Persistence Landscape", ax=None):
    """
    Plot persistence landscape
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Reshape if needed (assuming 5 layers)
    n_layers = 5
    if landscape.ndim == 1:
        n_samples = len(landscape) // n_layers
        if n_samples > 0:
            landscape_reshaped = landscape.reshape(n_layers, n_samples)
        else:
            landscape_reshaped = landscape.reshape(n_layers, 1)
    else:
        landscape_reshaped = landscape
    
    x = np.arange(landscape_reshaped.shape[1])
    
    for i in range(min(landscape_reshaped.shape[0], n_layers)):
        ax.plot(x, landscape_reshaped[i], label=f'Layer {i+1}', 
               alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Sample Index', fontsize=8)
    ax.set_ylabel('Landscape Value', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    if landscape_reshaped.shape[0] <= 5:
        ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def process_image_with_filtrations(image_path, image_name, output_dir):
    """
    Process a single image with all 4 filtrations and create visualization
    
    Args:
        image_path: Path to input image
        image_name: Name for output file
        output_dir: Directory to save results
    """
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required for this script")
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
    
    # Store diagrams, landscapes, and barcodes for each filtration
    filtration_names = ['height', 'radial', 'density', 'dilation']
    diagrams_dict = {}
    landscapes_dict = {}
    
    # Compute diagrams and landscapes for each filtration
    for filt_name in filtration_names:
        if filt_name not in filtered_grids:
            continue
        
        grid = filtered_grids[filt_name]
        grid_norm = normalize_image(grid)
        
        # Compute persistence diagram
        persistence.images = grid_norm
        diagrams = persistence.compute_diagram()
        diagrams_dict[filt_name] = diagrams[0]  # Single image
        
        # Compute persistence landscape
        landscapes = persistence.compute_landscape(diagrams)
        landscapes_dict[filt_name] = landscapes[0]  # Single image
    
    # Create visualization with 3 rows: diagrams, barcodes, landscapes
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3,
                           width_ratios=[1, 1, 1, 1, 1])
    
    # Row 1: Original image + 4 persistence diagrams
    # Original image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title('Original Image', fontsize=10, fontweight='bold')
    ax_img.axis('off')
    
    # Persistence diagrams for each filtration
    for idx, filt_name in enumerate(filtration_names):
        if filt_name in diagrams_dict:
            ax_diag = fig.add_subplot(gs[0, idx + 1])
            plot_persistence_diagram(
                diagrams_dict[filt_name],
                title=f'{filt_name.capitalize()} Filtration',
                ax=ax_diag
            )
    
    # Row 2: Empty space + 4 barcodes
    # Empty space (for alignment)
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')
    
    # Barcodes for each filtration
    for idx, filt_name in enumerate(filtration_names):
        if filt_name in diagrams_dict:
            ax_barcode = fig.add_subplot(gs[1, idx + 1])
            plot_barcode(
                diagrams_dict[filt_name],
                title=f'{filt_name.capitalize()} Barcode',
                ax=ax_barcode
            )
    
    # Row 3: Empty space + 4 persistence landscapes
    # Empty space (for alignment)
    ax_empty2 = fig.add_subplot(gs[2, 0])
    ax_empty2.axis('off')
    
    # Persistence landscapes for each filtration
    for idx, filt_name in enumerate(filtration_names):
        if filt_name in landscapes_dict:
            ax_landscape = fig.add_subplot(gs[2, idx + 1])
            plot_persistence_landscape(
                landscapes_dict[filt_name],
                title=f'{filt_name.capitalize()} Landscape',
                ax=ax_landscape
            )
    
    # Add overall title
    plt.suptitle(f'Multi-Filtration Persistence Analysis: {image_name}', 
                fontsize=14, fontweight='bold', y=0.99)
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_name}_multi_filtration.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")
    return diagrams_dict


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate persistence diagrams and barcodes for 40 images with 4 filtrations'
    )
    parser.add_argument('--real-dir', type=str, 
                       default='../preprocessing_raw_image_data/val_gray/real',
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/fake',
                       help='Directory containing fake images')
    parser.add_argument('-o', '--output', type=str,
                       default='Multi_Filtration_Results',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Maximum number of images per class (default: 20)')
    
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
    for i, img_file in enumerate(real_images, 1):
        print(f"[{i}/{len(real_images)}] Processing: {img_file.name}")
        process_image_with_filtrations(
            img_file,
            f"real_{img_file.stem}",
            output_dir
        )
    
    # Process fake images
    print("\nProcessing FAKE images...")
    for i, img_file in enumerate(fake_images, 1):
        print(f"[{i}/{len(fake_images)}] Processing: {img_file.name}")
        process_image_with_filtrations(
            img_file,
            f"fake_{img_file.stem}",
            output_dir
        )
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Total images processed: {len(real_images) + len(fake_images)}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

