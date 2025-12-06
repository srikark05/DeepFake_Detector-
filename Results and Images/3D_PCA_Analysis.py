#!/usr/bin/env python3
"""
3D PCA Analysis for Persistence Landscapes and Homology Features
Performs 3D Principal Component Analysis to visualize separability
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Homology and Topology"))

try:
    from T_Diagrams import Persistence
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
        # Resize for consistency
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


def extract_persistence_landscapes(image_paths, labels):
    """
    Extract persistence landscapes for a batch of images
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (0=fake, 1=real)
    
    Returns:
        landscapes: Array of persistence landscapes
        valid_indices: Indices of successfully processed images
    """
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required")
        return None, []
    
    landscapes = []
    valid_indices = []
    
    print("Extracting persistence landscapes...")
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"  Processing image {i+1}/{len(image_paths)}...")
        
        try:
            image = load_image(img_path)
            if image is None:
                continue
            
            image_norm = normalize_image(image)
            
            # Initialize persistence module
            persistence = Persistence(image_norm)
            
            # Compute persistence diagram
            diagrams = persistence.compute_diagram()
            
            # Compute persistence landscape
            landscape = persistence.compute_landscape(diagrams)
            
            # Flatten landscape
            landscape_flat = landscape[0].flatten()
            
            landscapes.append(landscape_flat)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"  Warning: Failed to process {img_path}: {e}")
            continue
    
    if len(landscapes) == 0:
        return None, []
    
    landscapes_array = np.vstack(landscapes)
    return landscapes_array, valid_indices


def compute_homology_statistics(diagram):
    """
    Compute statistics for each homology dimension
    
    Args:
        diagram: Persistence diagram (n_points, 3) with [dim, birth, death]
    
    Returns:
        stats: Dictionary with statistics for H0, H1, H2
    """
    separated = {}
    for dim in [0, 1, 2]:
        dim_points = diagram[diagram[:, 0] == dim]
        if len(dim_points) > 0:
            births = dim_points[:, 1]
            deaths = dim_points[:, 2]
            persistences = deaths - births
            persistences = persistences[persistences > 0]
            
            separated[dim] = {
                'count': len(persistences),
                'mean_persistence': np.mean(persistences) if len(persistences) > 0 else 0,
                'max_persistence': np.max(persistences) if len(persistences) > 0 else 0,
                'total_persistence': np.sum(persistences) if len(persistences) > 0 else 0,
            }
        else:
            separated[dim] = {
                'count': 0,
                'mean_persistence': 0,
                'max_persistence': 0,
                'total_persistence': 0,
            }
    
    return separated


def analyze_homology_features(image_paths, labels):
    """
    Analyze homology features for real and fake images
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (0=fake, 1=real)
    
    Returns:
        features: Array of homology features
        valid_indices: Indices of successfully processed images
    """
    if not GTDA_AVAILABLE:
        print("Error: giotto-tda is required")
        return None, []
    
    features_list = []
    valid_indices = []
    
    print("Computing homology features...")
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"  Processing image {i+1}/{len(image_paths)}...")
        
        try:
            image = load_image(img_path)
            if image is None:
                continue
            
            image_norm = normalize_image(image)
            
            # Initialize persistence module
            persistence = Persistence(image_norm)
            
            # Compute persistence diagram
            diagrams = persistence.compute_diagram()
            diagram = diagrams[0]
            
            # Compute statistics
            stats = compute_homology_statistics(diagram)
            
            # Create feature vector
            feature_vector = []
            for dim in [0, 1, 2]:
                feature_vector.extend([
                    stats[dim]['count'],
                    stats[dim]['mean_persistence'],
                    stats[dim]['max_persistence'],
                    stats[dim]['total_persistence'],
                ])
            
            features_list.append(feature_vector)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"  Warning: Failed to process {img_path}: {e}")
            continue
    
    if len(features_list) == 0:
        return None, []
    
    features_array = np.array(features_list)
    return features_array, valid_indices


def plot_3d_pca(data, labels, valid_indices, output_path, title_suffix=""):
    """
    Perform 3D PCA and visualize
    
    Args:
        data: Array of features (landscapes or homology features)
        labels: List of labels (0=fake, 1=real)
        valid_indices: Indices of valid samples
        output_path: Path to save the plot
        title_suffix: Additional text for title
    """
    # Filter labels for valid indices
    valid_labels = [labels[i] for i in valid_indices]
    valid_labels = np.array(valid_labels)
    
    # Standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform 3D PCA
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    # Separate by class
    real_mask = valid_labels == 1
    fake_mask = valid_labels == 0
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if np.any(real_mask):
        ax.scatter(data_pca[real_mask, 0], data_pca[real_mask, 1], data_pca[real_mask, 2],
                  c='green', label='Real', alpha=0.6, s=50, edgecolors='darkgreen', linewidths=0.5)
    
    if np.any(fake_mask):
        ax.scatter(data_pca[fake_mask, 0], data_pca[fake_mask, 1], data_pca[fake_mask, 2],
                  c='red', label='Fake', alpha=0.6, s=50, edgecolors='darkred', linewidths=0.5)
    
    # Set labels
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', 
                 fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', 
                 fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}% variance)', 
                 fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_title(f'3D PCA Analysis{title_suffix}\n'
                f'Total Explained Variance: {total_variance*100:.1f}%',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='upper left')
    
    # Add statistics text
    stats_text = f'Real: {np.sum(real_mask)}, Fake: {np.sum(fake_mask)}\n'
    stats_text += f'PC1: {explained_variance[0]*100:.2f}%\n'
    stats_text += f'PC2: {explained_variance[1]*100:.2f}%\n'
    stats_text += f'PC3: {explained_variance[2]*100:.2f}%'
    
    # Position text box
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 3D PCA plot: {output_path}")
    print(f"  Explained variance: PC1={explained_variance[0]*100:.2f}%, "
          f"PC2={explained_variance[1]*100:.2f}%, "
          f"PC3={explained_variance[2]*100:.2f}%, "
          f"Total={total_variance*100:.2f}%")
    
    # Calculate separability metric (silhouette score)
    if len(np.unique(valid_labels)) > 1:
        silhouette = silhouette_score(data_pca, valid_labels)
        print(f"  Silhouette score: {silhouette:.4f} "
              f"({'Good separation' if silhouette > 0.5 else 'Moderate separation' if silhouette > 0.3 else 'Poor separation'})")
    
    return data_pca, explained_variance


def create_multiple_views(data_pca, labels, valid_indices, output_dir, base_name):
    """
    Create multiple 3D views from different angles
    
    Args:
        data_pca: 3D PCA transformed data
        labels: List of labels
        valid_indices: Valid sample indices
        output_dir: Output directory
        base_name: Base name for output files
    """
    valid_labels = np.array([labels[i] for i in valid_indices])
    real_mask = valid_labels == 1
    fake_mask = valid_labels == 0
    
    # Different viewing angles
    views = [
        (20, 45, "standard"),
        (20, 135, "side"),
        (90, 0, "top"),
        (0, 0, "front"),
        (20, 225, "back_side"),
    ]
    
    for elev, azim, view_name in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if np.any(real_mask):
            ax.scatter(data_pca[real_mask, 0], data_pca[real_mask, 1], data_pca[real_mask, 2],
                      c='green', label='Real', alpha=0.6, s=50, edgecolors='darkgreen', linewidths=0.5)
        
        if np.any(fake_mask):
            ax.scatter(data_pca[fake_mask, 0], data_pca[fake_mask, 1], data_pca[fake_mask, 2],
                      c='red', label='Fake', alpha=0.6, s=50, edgecolors='darkred', linewidths=0.5)
        
        ax.set_xlabel('PC1', fontsize=11, labelpad=8)
        ax.set_ylabel('PC2', fontsize=11, labelpad=8)
        ax.set_zlabel('PC3', fontsize=11, labelpad=8)
        ax.set_title(f'3D PCA - {view_name.replace("_", " ").title()} View', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        
        output_file = Path(output_dir) / f"{base_name}_{view_name}_view.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {view_name} view: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='3D PCA analysis for persistence landscapes and homology features'
    )
    parser.add_argument('--real-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/real',
                       help='Directory containing real images')
    parser.add_argument('--fake-dir', type=str,
                       default='../preprocessing_raw_image_data/val_gray/fake',
                       help='Directory containing fake images')
    parser.add_argument('-o', '--output', type=str,
                       default='Homology_PCA_Results',
                       help='Output directory')
    parser.add_argument('-m', '--max-images', type=int, default=100,
                       help='Maximum number of images per class')
    parser.add_argument('--views', action='store_true',
                       help='Generate multiple viewing angles')
    
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
    
    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    fake_images = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
    
    if args.max_images:
        real_images = real_images[:args.max_images]
        fake_images = fake_images[:args.max_images]
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Prepare data
    all_images = real_images + fake_images
    all_labels = [1] * len(real_images) + [0] * len(fake_images)
    
    # Extract persistence landscapes
    print("\n" + "="*60)
    print("Step 1: Extracting Persistence Landscapes")
    print("="*60)
    landscapes, valid_indices = extract_persistence_landscapes(all_images, all_labels)
    
    if landscapes is None or len(landscapes) == 0:
        print("Error: Failed to extract landscapes")
        return
    
    print(f"Successfully extracted landscapes for {len(landscapes)} images")
    
    # Perform 3D PCA on landscapes
    print("\n" + "="*60)
    print("Step 2: Performing 3D PCA on Persistence Landscapes")
    print("="*60)
    pca_output = output_dir / "3D_PCA_Persistence_Landscapes.png"
    landscapes_pca, explained_variance = plot_3d_pca(
        landscapes, all_labels, valid_indices, pca_output,
        " - Persistence Landscapes"
    )
    
    # Generate multiple views if requested
    if args.views:
        print("\nGenerating multiple viewing angles...")
        create_multiple_views(landscapes_pca, all_labels, valid_indices, 
                            output_dir, "3D_PCA_Persistence_Landscapes")
    
    # Analyze homology features
    print("\n" + "="*60)
    print("Step 3: Analyzing Homology Features")
    print("="*60)
    features, feature_valid_indices = analyze_homology_features(all_images, all_labels)
    
    if features is not None and len(features) > 0:
        # Perform 3D PCA on homology features
        print("\n" + "="*60)
        print("Step 4: 3D PCA on Homology Features")
        print("="*60)
        features_pca_output = output_dir / "3D_PCA_Homology_Features.png"
        features_pca, feat_explained = plot_3d_pca(
            features, all_labels, feature_valid_indices, 
            features_pca_output, " - Homology Features"
        )
        
        # Generate multiple views if requested
        if args.views:
            print("\nGenerating multiple viewing angles for homology features...")
            create_multiple_views(features_pca, all_labels, feature_valid_indices,
                                output_dir, "3D_PCA_Homology_Features")
    
    print("\n" + "="*60)
    print("3D PCA Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

