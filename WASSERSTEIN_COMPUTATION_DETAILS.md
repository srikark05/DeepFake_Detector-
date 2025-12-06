# Wasserstein Amplitude Computation: Detailed Explanation

## Answer: Diagram to Diagonal (Not Diagram-to-Diagram)

The Wasserstein amplitude is computed as **diagram to diagonal** (empty diagram), **NOT diagram-to-diagram**.

For each filtration step, we compute:
- **One persistence diagram** from the filtered image
- **One Wasserstein amplitude** = distance from that diagram to the empty diagram (diagonal)

## Detailed Flow

### Step-by-Step Process

```
For each image:
  ├── Apply Height Filtration
  │   ├── Compute Persistence Diagram (D_height)
  │   └── Compute Wasserstein Amplitude: W(D_height, Δ)  ← Diagram to Diagonal
  │
  ├── Apply Radial Filtration
  │   ├── Compute Persistence Diagram (D_radial)
  │   └── Compute Wasserstein Amplitude: W(D_radial, Δ)  ← Diagram to Diagonal
  │
  ├── Apply Density Filtration
  │   ├── Compute Persistence Diagram (D_density)
  │   └── Compute Wasserstein Amplitude: W(D_density, Δ)  ← Diagram to Diagonal
  │
  └── Apply Dilation Filtration
      ├── Compute Persistence Diagram (D_dilation)
      └── Compute Wasserstein Amplitude: W(D_dilation, Δ)  ← Diagram to Diagonal
```

### Code Flow

**In `pipeline.py` (lines 244-262):**

```python
for filt_name, grid in filtered_grids.items():
    # 1. Normalize the filtered image
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)
    
    # 2. Compute ONE persistence diagram for this filtration
    persistence_module.images = grid_norm
    diagrams = persistence_module.compute_diagram()  # Single diagram
    
    # 3. Extract features (including Wasserstein amplitude)
    feature_extractor = TDA_FeatureExtractor()
    feats = feature_extractor.extract_features(diagrams, persistence_module)
    # This computes: W(diagrams, Δ) for THIS diagram
```

**In `Entropy.py` (line 26):**

```python
# 3. Wasserstein amplitude
A = persistence_module.compute_amplitude(diagrams)
# This computes: W(diagrams, Δ)
# NOT: W(diagram1, diagram2)
```

**In `T_Diagrams.py` (line 23, 59-63):**

```python
self.amplitude = Amplitude(metric="wasserstein")
# This is initialized once and reused

def compute_amplitude(self, diagrams):
    """
    Wasserstein amplitude (scalar or vector).
    Computes: W(diagrams, Δ) where Δ is the diagonal (empty diagram)
    """
    return self.amplitude.fit_transform(diagrams)
```

## What is the "Diagonal" (Δ)?

The **diagonal** (Δ) represents the **empty persistence diagram** - a diagram with no topological features. It's the set of all points on the line `y = x` (birth = death).

- **Empty diagram** = No topological features
- **Wasserstein amplitude** = Distance from your diagram to this empty state
- **Higher amplitude** = More significant topological features

## Mathematical Definition

For a persistence diagram **D**, the Wasserstein amplitude is:

```
A_p(D) = W_p(D, Δ)
```

Where:
- `W_p` = p-Wasserstein distance (typically p=2)
- `D` = Your persistence diagram (from one filtration)
- `Δ` = Diagonal (empty diagram)
- **NOT** comparing two different diagrams to each other

## Why Not Diagram-to-Diagram?

If we were doing diagram-to-diagram comparison, we would need:
- Compare height diagram vs radial diagram
- Compare height diagram vs density diagram
- etc.

But that's **NOT** what we're doing. Instead:

- **Height filtration** → Diagram → Amplitude (distance to empty)
- **Radial filtration** → Diagram → Amplitude (distance to empty)
- **Density filtration** → Diagram → Amplitude (distance to empty)
- **Dilation filtration** → Diagram → Amplitude (distance to empty)

Each amplitude is computed **independently** for each filtration's diagram.

## Output Structure

For a single image, you get **4 Wasserstein amplitudes** (one per filtration):

```
Feature Vector = [
    Landscapes (from 4 filtrations),
    Entropy (from 4 filtrations),
    Wasserstein Amplitudes (from 4 filtrations),  ← 4 values
    Betti Curves (from 4 filtrations)
]
```

Each amplitude value tells you: "How topologically complex is this filtered image?"

## Example

Consider processing one image:

1. **Height Filtration**:
   - Creates filtered image → Persistence diagram D₁
   - Amplitude: A₁ = W(D₁, Δ) = 0.45

2. **Radial Filtration**:
   - Creates filtered image → Persistence diagram D₂
   - Amplitude: A₂ = W(D₂, Δ) = 0.32

3. **Density Filtration**:
   - Creates filtered image → Persistence diagram D₃
   - Amplitude: A₃ = W(D₃, Δ) = 0.67

4. **Dilation Filtration**:
   - Creates filtered image → Persistence diagram D₄
   - Amplitude: A₄ = W(D₄, Δ) = 0.28

**We do NOT compute:**
- ❌ W(D₁, D₂) - comparing height to radial
- ❌ W(D₁, D₃) - comparing height to density
- ❌ etc.

**We DO compute:**
- ✅ W(D₁, Δ) - height diagram complexity
- ✅ W(D₂, Δ) - radial diagram complexity
- ✅ W(D₃, Δ) - density diagram complexity
- ✅ W(D₄, Δ) - dilation diagram complexity

## Why This Approach?

1. **Independent Features**: Each filtration reveals different aspects
2. **Scalable**: O(n) complexity per diagram, not O(n²) for comparisons
3. **Interpretable**: Each amplitude measures complexity of one filtration
4. **Stable**: Distance to diagonal is stable under small perturbations

## Summary

**Question**: Are we computing Wasserstein amplitude for each filtration step in the diagram, or diagram-to-diagram?

**Answer**: **For each filtration step, we compute one persistence diagram, then compute the Wasserstein amplitude from that diagram to the empty diagram (diagonal).**

- ✅ **Diagram → Diagonal** (what we do)
- ❌ **Diagram → Diagram** (what we don't do)

Each filtration produces its own independent amplitude value, measuring the topological complexity of that particular filtered view of the image.

