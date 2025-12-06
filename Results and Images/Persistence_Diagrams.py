#!/usr/bin/env python3
"""
Persistence Diagrams, Barcodes, and Landscapes Generator
Generates persistence diagrams (H0, H1, H2), barcodes, and persistence landscapes
for real and fake images
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Homology and Topology"))

try:
    from T_Diagrams import Persistence
    from gtda.plotting import plot_diagram
    GTDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: giotto-tda not available: {e}")
    print("This script requires giotto-tda to be installed.")
    GTDA_AVAILABLE = False


def load_image(image_path):
    """Load a single image and convert to numpy array"""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        # Resize for performance
        img = img.resize((128, 128), Image.LANCZOS)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


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
    
    Args:
        diagram: Array of shape (n_points, 3) with [dim, birth, death]
        title: Plot title
        ax: Matplotlib axis (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    separated = separate_diagrams_by_dimension(diagram)
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}
    
    for dim in [0, 1, 2]:
        points = separated[dim]
        if len(points) > 0:
            births = points[:, 0]
            deaths = points[:, 1]
            ax.scatter(births, deaths, c=colors[dim], label=labels[dim], 
                      alpha=0.6, s=30)
    
    # Plot diagonal line
    max_val = max(diagram[:, 1:].max(), 1.0) if len(diagram) > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_barcode(diagram, title="Persistence Barcode", ax=None):
    """
    Plot persistence barcode for each homology dimension
    
    Args:
        diagram: Array of shape (n_points, 3) with [dim, birth, death]
        title: Plot title
        ax: Matplotlib axis (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    separated = separate_diagrams_by_dimension(diagram)
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}
    
    y_pos = 0
    y_ticks = []
    y_labels = []
    
    for dim in [0, 1, 2]:
        points = separated[dim]
        if len(points) > 0:
            for i, (birth, death) in enumerate(points):
                # Only plot if persistence > 0
                if death > birth:
                    ax.plot([birth, death], [y_pos, y_pos], 
                           color=colors[dim], linewidth=2, alpha=0.7)
                    y_pos += 1
            
            if len(points) > 0:
                y_ticks.append(y_pos - len(points[points[:, 1] > points[:, 0]]) / 2)
                y_labels.append(labels[dim])
                y_pos += 2  # Space between dimensions
    
    ax.set_xlabel('Filtration Value', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_persistence_landscape(landscape, title="Persistence Landscape", ax=None):
    """
    Plot persistence landscape
    
    Args:
        landscape: Array of shape (n_layers * n_samples,)
        title: Plot title
        ax: Matplotlib axis (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Reshape if needed (assuming 5 layers)
    n_layers = 5
    if landscape.ndim == 1:
        n_samples = len(landscape) // n_layers
        landscape_reshaped = landscape.reshape(n_layers, n_samples)
    else:
        landscape_reshaped = landscape
    
    x = np.arange(landscape_reshaped.shape[1])
    
    for i in range(landscape_reshaped.shape[0]):
        ax.plot(x, landscape_reshaped[i], label=f'Layer {i+1}', 
               alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Landscape Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def process_image_for_persistence(image_path, output_dir, image_name):
    """
    Process a single image and generate all persistence visualizations
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        image_name: Base name for output files
    """
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required for this script")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and normalize image
    print(f"Processing: {image_name}")
    image = load_image(image_path)
    if image is None:
        return None
    
    image_norm = normalize_image(image)
    
    # Initialize persistence module
    persistence = Persistence(image_norm)
    
    # Compute persistence diagram
    diagrams = persistence.compute_diagram()
    diagram = diagrams[0]  # Single image
    
    # Compute persistence landscape
    landscapes = persistence.compute_landscape(diagrams)
    landscape = landscapes[0]  # Single image
    
    # Separate by dimension
    separated = separate_diagrams_by_dimension(diagram)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Persistence diagram
    ax2 = fig.add_subplot(gs[0, 1:])
    plot_persistence_diagram(diagram, "Persistence Diagram (H0, H1, H2)", ax2)
    
    # Barcode
    ax3 = fig.add_subplot(gs[1, :])
    plot_barcode(diagram, "Persistence Barcode", ax3)
    
    # Persistence landscape
    ax4 = fig.add_subplot(gs[2, :])
    plot_persistence_landscape(landscape, "Persistence Landscape", ax4)
    
    plt.suptitle(f'Persistence Analysis: {image_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save combined figure
    output_file = output_path / f"{image_name}_persistence_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved combined analysis: {output_file}")
    
    # Save individual diagrams by dimension
    for dim in [0, 1, 2]:
        dim_points = separated[dim]
        if len(dim_points) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            colors = {0: 'blue', 1: 'red', 2: 'green'}
            labels = {0: 'H0', 1: 'H1', 2: 'H2'}
            
            births = dim_points[:, 0]
            deaths = dim_points[:, 1]
            ax.scatter(births, deaths, c=colors[dim], alpha=0.6, s=50)
            
            max_val = max(deaths.max(), births.max(), 1.0)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel('Birth', fontsize=12)
            ax.set_ylabel('Death', fontsize=12)
            ax.set_title(f'{labels[dim]} Persistence Diagram: {image_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            output_file = output_path / f"{image_name}_H{dim}_diagram.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved H{dim} diagram: {output_file}")
    
    # Save barcode separately
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plot_barcode(diagram, f"Persistence Barcode: {image_name}", ax)
    output_file = output_path / f"{image_name}_barcode.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved barcode: {output_file}")
    
    # Save landscape separately
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    plot_persistence_landscape(landscape, f"Persistence Landscape: {image_name}", ax)
    output_file = output_path / f"{image_name}_landscape.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved landscape: {output_file}")
    
    return {
        'diagram': diagram,
        'landscape': landscape,
        'separated': separated
    }


def process_batch_images(image_dir, label, output_dir, max_images=10):
    """
    Process multiple images from a directory
    
    Args:
        image_dir: Directory containing images
        label: Label for the images ('real' or 'fake')
        output_dir: Directory to save results
        max_images: Maximum number of images to process
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return []
    
    # Find all image files
    image_files = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\nProcessing {len(image_files)} {label} images...")
    
    results = []
    output_subdir = Path(output_dir) / label
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        result = process_image_for_persistence(
            img_file, 
            output_subdir, 
            f"{label}_{img_file.stem}"
        )
        if result:
            result['label'] = label
            result['image_path'] = str(img_file)
            results.append(result)
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate persistence diagrams, barcodes, and landscapes'
    )
    parser.add_argument('--real-dir', type=str, 
                       default='../preprocessing_raw_image_data/val_gray/real',
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/fake',
                       help='Directory containing fake images')
    parser.add_argument('-o', '--output', type=str,
                       default='Persistence_Results',
                       help='Output directory')
    parser.add_argument('-m', '--max-images', type=int, default=5,
                       help='Maximum number of images per class')
    
    args = parser.parse_args()
    
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required. Please install it first.")
        print("See INSTALL_GIOTTO.md for instructions.")
        sys.exit(1)
    
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process real images
    real_results = process_batch_images(
        args.real_dir, 'real', output_dir, args.max_images
    )
    
    # Process fake images
    fake_results = process_batch_images(
        args.fake_dir, 'fake', output_dir, args.max_images
    )
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Real images processed: {len(real_results)}")
    print(f"Fake images processed: {len(fake_results)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

