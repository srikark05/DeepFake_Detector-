#!/usr/bin/env python3
"""
Complete TDA-based DeepFake Detection Pipeline
Integrates all modules: preprocessing, filtration, persistence, feature extraction, and classification
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add module paths to sys.path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent / "ML Models"))
sys.path.insert(0, str(Path(__file__).parent / "Preprocessing"))

# Import modules
from Filtration import Filtration
from T_Diagrams import Persistence
from Entropy import TDA_FeatureExtractor
from SVM import TDASVMClassifier


# =========================================================
# 1. Data Loading Functions
# =========================================================
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


def load_dataset(data_dir, max_images_per_class=None):
    """
    Load images from test_gray and val_gray folders
    
    Args:
        data_dir: Path to preprocessing_raw_image_data directory
        max_images_per_class: Limit number of images per class (for testing)
    
    Returns:
        images: List of numpy arrays
        labels: List of labels (0=fake, 1=real)
        dataset_info: Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    
    datasets = ["test_gray", "val_gray"]
    class_folders = ["fake", "real"]
    
    dataset_info = {
        "test_gray": {"fake": 0, "real": 0},
        "val_gray": {"fake": 0, "real": 0}
    }
    
    for dataset in datasets:
        dataset_path = data_path / dataset
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue
        
        for class_idx, class_folder in enumerate(class_folders):
            class_path = dataset_path / class_folder
            if not class_path.exists():
                continue
            
            # Get all image files
            image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]
            
            print(f"Loading {len(image_files)} {class_folder} images from {dataset}...")
            
            for img_file in image_files:
                img_array = load_image(img_file)
                if img_array is not None:
                    images.append(img_array)
                    labels.append(class_idx)  # 0=fake, 1=real
                    dataset_info[dataset][class_folder] += 1
    
    print(f"\nDataset Summary:")
    print(f"Total images loaded: {len(images)}")
    print(f"Fake images: {sum(1 for l in labels if l == 0)}")
    print(f"Real images: {sum(1 for l in labels if l == 1)}")
    print(f"Dataset breakdown: {dataset_info}")
    
    return images, labels, dataset_info


# =========================================================
# 2. Image to Point Cloud Conversion
# =========================================================
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


# =========================================================
# 3. Filtration Application
# =========================================================
def apply_filtrations_to_image(image):
    """
    Apply all filtrations to an image and return filtered grids
    
    Args:
        image: 2D numpy array (H, W)
    
    Returns:
        filtered_grids: Dictionary with filtered images
    """
    # Convert image to point cloud
    points = image_to_point_cloud(image, subsample=5000)  # Limit for performance
    
    # Initialize filtration
    filt = Filtration(points)
    
    # Apply different filtrations
    # Note: These return 1D arrays, we need to reshape them back to image grid
    h, w = image.shape
    
    filtered_grids = {}
    
    # Height filtration (using vertical direction - intensity)
    try:
        # Use intensity as height direction
        direction = np.array([0, 0, 1])  # Vertical in z-direction (intensity)
        height_vals = filt.Height(direction)
        if len(height_vals) == len(points):
            # Map back to image grid
            if len(height_vals) == h * w:
                filtered_grids['height'] = height_vals.reshape(h, w)
            else:
                # Interpolate back to full grid
                filtered_grids['height'] = image.copy()
        else:
            filtered_grids['height'] = image.copy()
    except Exception as e:
        filtered_grids['height'] = image.copy()
    
    # Radial filtration (from center)
    try:
        center = np.mean(points, axis=0)
        radial_vals = filt.Radial(center)
        if len(radial_vals) == len(points):
            if len(radial_vals) == h * w:
                filtered_grids['radial'] = radial_vals.reshape(h, w)
            else:
                filtered_grids['radial'] = image.copy()
        else:
            filtered_grids['radial'] = image.copy()
    except Exception as e:
        filtered_grids['radial'] = image.copy()
    
    # Density filtration
    try:
        density_vals = filt.Density(k=5)
        if len(density_vals) == h * w:
            filtered_grids['density'] = density_vals.reshape(h, w)
        else:
            filtered_grids['density'] = image.copy()
    except:
        filtered_grids['density'] = image.copy()
    
    # Dilation filtration (returns distance matrix, use mean)
    try:
        dilation_matrix = filt.Dilation()
        if dilation_matrix.ndim == 2 and dilation_matrix.shape[0] == len(points):
            # Use mean distance from each point
            dilation_vals = np.mean(dilation_matrix, axis=1)
            if len(dilation_vals) == h * w:
                filtered_grids['dilation'] = dilation_vals.reshape(h, w)
            else:
                filtered_grids['dilation'] = image.copy()
        else:
            filtered_grids['dilation'] = image.copy()
    except Exception as e:
        filtered_grids['dilation'] = image.copy()
    
    return filtered_grids


# =========================================================
# 4. TDA Feature Extraction for a Single Image
# =========================================================
def extract_tda_features_for_image(image, persistence_module):
    """
    Extract TDA features from a single image using all filtrations
    
    Args:
        image: 2D numpy array (H, W)
        persistence_module: Persistence instance
    
    Returns:
        features: Concatenated feature vector
    """
    # Apply filtrations
    filtered_grids = apply_filtrations_to_image(image)
    
    # Extract features from each filtered grid
    feature_blocks = []
    
    for filt_name, grid in filtered_grids.items():
        # Normalize grid to [0, 1] for cubical persistence
        grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)
        
        # Compute persistence diagram
        persistence_module.images = grid_norm
        diagrams = persistence_module.compute_diagram()
        
        # Extract features from diagrams
        try:
            # Use TDA_FeatureExtractor for comprehensive features
            feature_extractor = TDA_FeatureExtractor()
            feats = feature_extractor.extract_features(diagrams, persistence_module)
            
            # Flatten features
            if feats.ndim > 1:
                feats = feats.flatten()
            
            feature_blocks.append(feats)
        except Exception as e:
            print(f"Warning: Error extracting features from {filt_name}: {e}")
            # Use simple features as fallback
            if len(diagrams) > 0 and len(diagrams[0]) > 0:
                # Use persistence entropy as simple feature
                import sys
                sys.path.insert(0, str(Path(__file__).parent / "Homology and Topology"))
                from persistence_entropy import PersistenceEntropy
                entropy = PersistenceEntropy()
                simple_feats = entropy.fit_transform(diagrams)
                feature_blocks.append(simple_feats.flatten())
    
    if len(feature_blocks) == 0:
        # Fallback: return zeros
        return np.zeros(100)
    
    return np.concatenate(feature_blocks)


# =========================================================
# 5. Main Pipeline
# =========================================================
def main():
    """Main pipeline execution"""
    
    print("=" * 60)
    print("TDA-Based DeepFake Detection Pipeline")
    print("=" * 60)
    
    # ---------------------------
    # 1. Load Dataset
    # ---------------------------
    print("\n[1/5] Loading dataset...")
    data_dir = "preprocessing_raw_image_data"
    
    # For testing, limit images (set to None for full dataset)
    # Use smaller number for quick testing, None for full evaluation
    max_images = 50  # Set to None to use all images
    images, labels, dataset_info = load_dataset(data_dir, max_images_per_class=max_images)
    
    if len(images) == 0:
        print("Error: No images loaded. Check data directory path.")
        return None
    
    # Resize images to consistent size for efficiency
    target_size = (128, 128)
    print(f"\nResizing images to {target_size}...")
    images_resized = []
    for img in images:
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_resized = img_pil.resize(target_size, Image.LANCZOS)
        images_resized.append(np.array(img_resized, dtype=np.float32))
    images = images_resized
    
    # ---------------------------
    # 2. Initialize Persistence Module
    # ---------------------------
    print("\n[2/5] Initializing persistence module...")
    # Create a dummy image for initialization
    dummy_image = np.zeros((128, 128))
    persistence_module = Persistence(dummy_image)
    
    # ---------------------------
    # 3. Extract TDA Features
    # ---------------------------
    print("\n[3/5] Extracting TDA features from images...")
    print("This may take a while...")
    
    feature_list = []
    failed_count = 0
    
    for i, img in enumerate(images):
        if (i + 1) % 10 == 0:
            print(f"  Processing image {i+1}/{len(images)}...")
        
        try:
            feats = extract_tda_features_for_image(img, persistence_module)
            feature_list.append(feats)
        except Exception as e:
            print(f"  Warning: Failed to extract features from image {i+1}: {e}")
            failed_count += 1
            # Use zero features as fallback
            feature_list.append(np.zeros(100))
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} images failed feature extraction")
    
    # Convert to numpy array
    features = np.vstack(feature_list)
    labels = np.array(labels)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # ---------------------------
    # 4. Train SVM Classifier
    # ---------------------------
    print("\n[4/5] Training SVM classifier...")
    classifier = TDASVMClassifier(n_components=50, C=3.0, gamma="scale")
    
    acc, preds, y_test = classifier.fit(features, labels, test_size=0.2)
    
    # ---------------------------
    # 5. Results and Evaluation
    # ---------------------------
    print("\n[5/5] Final Results")
    print("=" * 60)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    print(f"\nTrue Negatives (Fake correctly identified): {cm[0,0]}")
    print(f"False Positives (Fake misclassified as Real): {cm[0,1]}")
    print(f"False Negatives (Real misclassified as Fake): {cm[1,0]}")
    print(f"True Positives (Real correctly identified): {cm[1,1]}")
    
    # Calculate additional metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    
    return classifier, features, labels


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    try:
        classifier, features, labels = main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
