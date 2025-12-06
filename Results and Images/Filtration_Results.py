#!/usr/bin/env python3
"""
Filtration Results Generator
Applies filtrations to images and saves visualization diagrams
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Filtration"))

from Filtration import Filtration


def load_image(image_path):
    """Load a single image and convert to numpy array"""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def image_to_point_cloud(image, subsample=None):
    """
    Convert a 2D grayscale image to a 3D point cloud
    Each point is (x, y, intensity)
    
    Args:
        image: 2D numpy array (H, W)
        subsample: If not None, subsample points to reduce computation
    
    Returns:
        points: (N, 3) array of points
    """
    h, w = image.shape
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Flatten and combine with intensity
    points = np.column_stack([
        x_coords.flatten(),
        y_coords.flatten(),
        image.flatten()
    ])
    
    # Subsample if requested
    if subsample and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points = points[indices]
    
    return points


def apply_filtrations_to_image(image, subsample=5000):
    """
    Apply all filtrations to an image and return filtered grids
    
    Args:
        image: 2D numpy array (H, W)
        subsample: Number of points to subsample for filtration
    
    Returns:
        filtered_grids: Dictionary with filtered images
        original_shape: Original image shape (h, w)
    """
    h, w = image.shape
    original_shape = (h, w)
    
    # Convert image to point cloud
    points = image_to_point_cloud(image, subsample=subsample)
    
    # Initialize filtration
    filt = Filtration(points)
    
    filtered_grids = {}
    
    # Height filtration (using vertical direction - intensity)
    try:
        direction = np.array([0, 0, 1])  # Vertical in z-direction (intensity)
        height_vals = filt.Height(direction)
        if len(height_vals) == len(points):
            # Map back to image grid if possible
            if len(height_vals) == h * w:
                filtered_grids['height'] = height_vals.reshape(h, w)
            else:
                # Create a grid by interpolating
                filtered_grids['height'] = _interpolate_to_grid(height_vals, points, h, w)
        else:
            filtered_grids['height'] = image.copy()
    except Exception as e:
        print(f"Warning: Height filtration failed: {e}")
        filtered_grids['height'] = image.copy()
    
    # Radial filtration (from center)
    try:
        center = np.mean(points, axis=0)
        radial_vals = filt.Radial(center)
        if len(radial_vals) == len(points):
            if len(radial_vals) == h * w:
                filtered_grids['radial'] = radial_vals.reshape(h, w)
            else:
                filtered_grids['radial'] = _interpolate_to_grid(radial_vals, points, h, w)
        else:
            filtered_grids['radial'] = image.copy()
    except Exception as e:
        print(f"Warning: Radial filtration failed: {e}")
        filtered_grids['radial'] = image.copy()
    
    # Density filtration
    try:
        density_vals = filt.Density(k=5)
        if len(density_vals) == len(points):
            if len(density_vals) == h * w:
                filtered_grids['density'] = density_vals.reshape(h, w)
            else:
                filtered_grids['density'] = _interpolate_to_grid(density_vals, points, h, w)
        else:
            filtered_grids['density'] = image.copy()
    except Exception as e:
        print(f"Warning: Density filtration failed: {e}")
        filtered_grids['density'] = image.copy()
    
    # Dilation filtration (returns distance matrix, use mean)
    try:
        dilation_matrix = filt.Dilation()
        if dilation_matrix.ndim == 2 and dilation_matrix.shape[0] == len(points):
            # Use mean distance from each point
            dilation_vals = np.mean(dilation_matrix, axis=1)
            if len(dilation_vals) == len(points):
                if len(dilation_vals) == h * w:
                    filtered_grids['dilation'] = dilation_vals.reshape(h, w)
                else:
                    filtered_grids['dilation'] = _interpolate_to_grid(dilation_vals, points, h, w)
            else:
                filtered_grids['dilation'] = image.copy()
        else:
            filtered_grids['dilation'] = image.copy()
    except Exception as e:
        print(f"Warning: Dilation filtration failed: {e}")
        filtered_grids['dilation'] = image.copy()
    
    return filtered_grids, original_shape


def _interpolate_to_grid(values, points, h, w):
    """
    Interpolate point cloud values back to image grid
    Simple nearest neighbor interpolation
    """
    from scipy.spatial import cKDTree
    
    # Create grid coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    grid_coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    
    # Use only x, y coordinates from points for interpolation
    point_coords = points[:, :2]
    
    # Find nearest neighbors
    tree = cKDTree(point_coords)
    _, indices = tree.query(grid_coords, k=1)
    
    # Map values to grid
    grid_values = values[indices].reshape(h, w)
    return grid_values


def save_filtration_diagrams(image, filtered_grids, output_dir, image_name):
    """
    Save visualization diagrams of all filtrations
    
    Args:
        image: Original image (2D array)
        filtered_grids: Dictionary of filtered images
        output_dir: Directory to save diagrams
        image_name: Base name for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Normalize images for display
    def normalize_for_display(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Filtration Results: {image_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Height filtration
    if 'height' in filtered_grids:
        height_norm = normalize_for_display(filtered_grids['height'])
        im1 = axes[0, 1].imshow(height_norm, cmap='viridis')
        axes[0, 1].set_title('Height Filtration', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Radial filtration
    if 'radial' in filtered_grids:
        radial_norm = normalize_for_display(filtered_grids['radial'])
        im2 = axes[0, 2].imshow(radial_norm, cmap='plasma')
        axes[0, 2].set_title('Radial Filtration', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Density filtration
    if 'density' in filtered_grids:
        density_norm = normalize_for_display(filtered_grids['density'])
        im3 = axes[1, 0].imshow(density_norm, cmap='inferno')
        axes[1, 0].set_title('Density Filtration', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Dilation filtration
    if 'dilation' in filtered_grids:
        dilation_norm = normalize_for_display(filtered_grids['dilation'])
        im4 = axes[1, 1].imshow(dilation_norm, cmap='magma')
        axes[1, 1].set_title('Dilation Filtration', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Combined comparison
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Filtration Analysis\nComplete', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the combined diagram
    output_file = output_path / f"{image_name}_filtration_diagram.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved filtration diagram: {output_file}")
    
    # Also save individual filtration images
    for filt_name, filt_image in filtered_grids.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        filt_norm = normalize_for_display(filt_image)
        
        cmap = {
            'height': 'viridis',
            'radial': 'plasma',
            'density': 'inferno',
            'dilation': 'magma'
        }.get(filt_name, 'gray')
        
        im = ax.imshow(filt_norm, cmap=cmap)
        ax.set_title(f'{filt_name.capitalize()} Filtration', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        individual_file = output_path / f"{image_name}_{filt_name}_filtration.png"
        plt.savefig(individual_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {filt_name} filtration: {individual_file}")


def process_single_image(image_path, output_dir=None, subsample=5000):
    """
    Process a single image and generate filtration diagrams
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results (default: Filtration_Results)
        subsample: Number of points to subsample
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "Filtration_Results"
    else:
        output_dir = Path(output_dir)
    
    # Load image
    print(f"\nProcessing image: {image_path}")
    image = load_image(image_path)
    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return
    
    # Get image name for output files
    image_name = Path(image_path).stem
    
    # Apply filtrations
    print("Applying filtrations...")
    filtered_grids, original_shape = apply_filtrations_to_image(image, subsample=subsample)
    
    # Save diagrams
    print("Generating and saving diagrams...")
    save_filtration_diagrams(image, filtered_grids, output_dir, image_name)
    
    print(f"✓ Completed processing: {image_name}\n")


def process_batch_images(image_dir, output_dir=None, max_images=None, subsample=5000):
    """
    Process multiple images from a directory
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save results
        max_images: Maximum number of images to process (None for all)
        subsample: Number of points to subsample
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    # Find all image files
    image_files = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg")) + \
                  list(image_path.glob("*.jpeg"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        process_single_image(img_file, output_dir, subsample=subsample)


def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate filtration diagrams for images')
    parser.add_argument('input', type=str, help='Input image file or directory')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory (default: Filtration_Results)')
    parser.add_argument('-s', '--subsample', type=int, default=5000,
                       help='Number of points to subsample (default: 5000)')
    parser.add_argument('-m', '--max-images', type=int, default=None,
                       help='Maximum number of images to process (for directories)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "Filtration_Results"
    
    if input_path.is_file():
        # Process single image
        process_single_image(input_path, output_dir, args.subsample)
    elif input_path.is_dir():
        # Process directory
        process_batch_images(input_path, output_dir, args.max_images, args.subsample)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage if run directly without arguments
    if len(sys.argv) == 1:
        print("Filtration Results Generator")
        print("=" * 60)
        print("\nUsage examples:")
        print("  python Filtration_Results.py <image_path>")
        print("  python Filtration_Results.py <image_dir> -m 10")
        print("  python Filtration_Results.py <image_path> -o custom_output_dir")
        print("\nFor help: python Filtration_Results.py -h")
        print("\nExample: Processing a sample image...")
        
        # Try to process a sample image if available
        sample_dir = Path(__file__).parent.parent / "preprocessing_raw_image_data" / "val_gray" / "real"
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.png"))[:1]
            if sample_images:
                print(f"\nProcessing sample image: {sample_images[0].name}")
                process_single_image(sample_images[0])
    else:
        main()

