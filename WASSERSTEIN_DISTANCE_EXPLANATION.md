# Wasserstein Distance in the DeepFake Detection Project

## Overview

The **Wasserstein distance** (also known as the Earth Mover's Distance) is used in this project through the **Wasserstein amplitude** metric to extract topological features from persistence diagrams. This helps quantify the topological structure differences between real and fake images.

## Where It's Used

### 1. **Initialization in `T_Diagrams.py`**

```python
self.amplitude = Amplitude(metric="wasserstein")
```

The `Amplitude` class from `giotto-tda` is initialized with the Wasserstein metric. This is defined in:
- **File**: `Homology and Topology/T_Diagrams.py`
- **Line**: 23

### 2. **Feature Extraction in `Entropy.py`**

```python
# 3. Wasserstein amplitude
A = persistence_module.compute_amplitude(diagrams)
```

The Wasserstein amplitude is computed and included as part of the feature vector in:
- **File**: `Homology and Topology/Entropy.py`
- **Line**: 26

### 3. **Integration in the Pipeline**

The Wasserstein amplitude features are extracted for each filtration type (height, radial, density, dilation) and concatenated with other TDA features in:
- **File**: `pipeline.py`
- **Function**: `extract_tda_features_for_image()`

## What is Wasserstein Distance?

The **Wasserstein distance** (specifically the p-Wasserstein distance) measures the "cost" of transforming one probability distribution into another. In the context of persistence diagrams:

- **Persistence diagrams** represent topological features (connected components, holes, voids) as points in 2D space (birth, death)
- **Wasserstein distance** measures how "different" two persistence diagrams are
- It finds the optimal matching between points in two diagrams and computes the total "cost" to transform one into the other

## Wasserstein Amplitude

The **Wasserstein amplitude** is a scalar (or vector) that measures the "size" or "magnitude" of a persistence diagram by computing its distance to the empty diagram (diagonal).

### Mathematical Definition

For a persistence diagram D, the Wasserstein amplitude is:

```
A_p(D) = W_p(D, Δ)
```

Where:
- `W_p` is the p-Wasserstein distance (typically p=2)
- `Δ` is the diagonal (empty diagram)
- This measures how "far" the diagram is from having no topological features

### Why It's Useful

1. **Quantifies Topological Complexity**: Higher amplitude = more significant topological features
2. **Stable Metric**: Small changes in the image result in small changes in amplitude
3. **Discriminative**: Real and fake images may have different topological structures, leading to different amplitudes

## How It Works in This Project

### Step-by-Step Process

1. **Image Processing**:
   - Images are converted to grayscale
   - Multiple filtrations are applied (height, radial, density, dilation)

2. **Persistence Diagram Computation**:
   - For each filtered image, a cubical persistence diagram is computed
   - This captures H0 (connected components), H1 (holes), H2 (voids)

3. **Wasserstein Amplitude Extraction**:
   ```python
   diagrams = persistence_module.compute_diagram()
   amplitude = persistence_module.compute_amplitude(diagrams)
   ```
   - Computes the Wasserstein distance from each diagram to the empty diagram
   - Returns a scalar or vector (one value per homology dimension)

4. **Feature Vector Construction**:
   - Wasserstein amplitude is combined with:
     - Persistence landscapes
     - Persistence entropy
     - Betti curves
   - This creates a comprehensive feature vector for classification

### Example Feature Vector

For a single image with 4 filtrations, the feature vector includes:

```
[Persistence Landscapes (from 4 filtrations),
 Persistence Entropy (from 4 filtrations),
 Wasserstein Amplitudes (from 4 filtrations),  ← Here!
 Betti Curves (from 4 filtrations)]
```

## Why Wasserstein Distance for DeepFake Detection?

### 1. **Captures Topological Differences**

Real images and deepfakes may have different topological structures:
- **Real images**: Natural textures, organic patterns
- **Deepfakes**: Artifacts from generation, synthetic patterns
- Wasserstein distance quantifies these differences

### 2. **Robust to Noise**

Wasserstein distance is stable under small perturbations, making it robust to:
- Image compression artifacts
- Minor preprocessing variations
- Small geometric transformations

### 3. **Multi-Scale Analysis**

By applying filtrations first, we analyze topology at different scales:
- Each filtration reveals different aspects of the image structure
- Wasserstein amplitude captures the "strength" of topological features at each scale

## Implementation Details

### In `T_Diagrams.py`:

```python
class Persistence:
    def __init__(self, images):
        # ...
        self.amplitude = Amplitude(metric="wasserstein")
        # This uses the 2-Wasserstein distance by default
    
    def compute_amplitude(self, diagrams):
        """
        Wasserstein amplitude (scalar or vector).
        """
        return self.amplitude.fit_transform(diagrams)
```

### In `Entropy.py`:

```python
class TDA_FeatureExtractor:
    def extract_features(self, diagrams, persistence_module):
        # ...
        # 3. Wasserstein amplitude
        A = persistence_module.compute_amplitude(diagrams)
        # A is a vector with one amplitude per homology dimension
        # Shape: (n_samples, n_homology_dims) or (n_samples, 1)
        
        features = np.concatenate([L, E, A, B], axis=1)
        return features
```

## Output Format

The Wasserstein amplitude can be:
- **Scalar**: Single value representing overall topological complexity
- **Vector**: One value per homology dimension (H0, H1, H2)

In this project, it's typically a vector with values for each homology dimension, providing:
- H0 amplitude: Complexity of connected components
- H1 amplitude: Complexity of holes/loops
- H2 amplitude: Complexity of voids/cavities

## Advantages for Classification

1. **Dimensionality**: Provides compact representation of topological information
2. **Discriminative Power**: Different images → different amplitudes → better classification
3. **Interpretability**: Higher amplitude = more complex topology
4. **Stability**: Robust to small image variations

## Relationship to Other Features

The Wasserstein amplitude complements other TDA features:

| Feature | What It Captures | Relationship to Wasserstein |
|---------|-----------------|----------------------------|
| **Persistence Landscapes** | Shape of topological features | Landscapes are derived from diagrams; amplitude summarizes them |
| **Persistence Entropy** | Distribution of feature lifetimes | Both measure diagram properties, but from different perspectives |
| **Betti Curves** | Number of features over filtration | Amplitude measures "strength", Betti measures "count" |

Together, these features provide a comprehensive view of the topological structure, enabling the SVM classifier to distinguish between real and fake images.

## Summary

The Wasserstein distance is used in this project through the **Wasserstein amplitude** metric to:

1. **Quantify topological complexity** of images after filtration
2. **Extract discriminative features** that help distinguish real from fake images
3. **Provide stable, robust metrics** that are insensitive to small perturbations
4. **Complement other TDA features** (landscapes, entropy, Betti curves) in the feature vector

This makes it a crucial component of the TDA-based deepfake detection pipeline, contributing to the overall classification performance.

