# Wasserstein Distance Computation - Clear Explanation

## Answer: Same Image, Different Filtrations

The Wasserstein amplitude is computed **on the same image**, but **for each filtration separately**. 

**It is NOT comparing different images to each other.**
**It is NOT comparing different filtrations to each other.**

## The Process

### For ONE Image:

```
Original Image
    │
    ├── Height Filtration
    │   └── Filtered Image (height)
    │       └── Persistence Diagram D₁
    │           └── Wasserstein Amplitude = W(D₁, Δ)  ← Diagram to Empty
    │
    ├── Radial Filtration
    │   └── Filtered Image (radial)
    │       └── Persistence Diagram D₂
    │           └── Wasserstein Amplitude = W(D₂, Δ)  ← Diagram to Empty
    │
    ├── Density Filtration
    │   └── Filtered Image (density)
    │       └── Persistence Diagram D₃
    │           └── Wasserstein Amplitude = W(D₃, Δ)  ← Diagram to Empty
    │
    └── Dilation Filtration
        └── Filtered Image (dilation)
            └── Persistence Diagram D₄
                └── Wasserstein Amplitude = W(D₄, Δ)  ← Diagram to Empty
```

## What We Compute

For **each filtration of the same image**:
1. Apply the filtration → Get filtered image
2. Compute persistence diagram from that filtered image
3. Compute: **W(diagram, empty_diagram)**

## What We Do NOT Compute

❌ **NOT**: W(D₁, D₂) - Comparing height diagram to radial diagram  
❌ **NOT**: W(D₁, D₃) - Comparing height diagram to density diagram  
❌ **NOT**: W(Image_A, Image_B) - Comparing different images  
❌ **NOT**: W(D₁, D₂, D₃, D₄) - Comparing all filtrations together

## Example: One Image

Let's say we have **Image_001**:

1. **Height Filtration**:
   - Input: Image_001
   - Output: Height-filtered version of Image_001
   - Persistence Diagram: D_height
   - Wasserstein Amplitude: **W(D_height, empty)** = 0.45

2. **Radial Filtration**:
   - Input: Image_001 (same image!)
   - Output: Radial-filtered version of Image_001
   - Persistence Diagram: D_radial
   - Wasserstein Amplitude: **W(D_radial, empty)** = 0.32

3. **Density Filtration**:
   - Input: Image_001 (same image!)
   - Output: Density-filtered version of Image_001
   - Persistence Diagram: D_density
   - Wasserstein Amplitude: **W(D_density, empty)** = 0.67

4. **Dilation Filtration**:
   - Input: Image_001 (same image!)
   - Output: Dilation-filtered version of Image_001
   - Persistence Diagram: D_dilation
   - Wasserstein Amplitude: **W(D_dilation, empty)** = 0.28

**Result**: 4 Wasserstein amplitudes for the same image, one per filtration.

## Why This Approach?

Each filtration reveals **different topological aspects** of the same image:
- **Height**: Intensity-based structure
- **Radial**: Distance-from-center patterns
- **Density**: Local point density
- **Dilation**: Distance relationships

Each gives a **different perspective** on the image's topology, so we compute amplitude for each independently.

## Feature Vector for One Image

For Image_001, the feature vector includes:
```
[
  ... (landscapes from 4 filtrations),
  ... (entropy from 4 filtrations),
  W(D_height, empty),    ← From height filtration
  W(D_radial, empty),    ← From radial filtration
  W(D_density, empty),   ← From density filtration
  W(D_dilation, empty), ← From dilation filtration
  ... (Betti curves from 4 filtrations)
]
```

## Summary

**Question**: Is Wasserstein distance computed on the same image filtration or different?

**Answer**: 
- ✅ **Same image** (Image_001)
- ✅ **Different filtrations** (height, radial, density, dilation)
- ✅ Each filtration gets its own Wasserstein amplitude
- ✅ Each amplitude = distance from that filtration's diagram to empty diagram
- ❌ **NOT** comparing filtrations to each other
- ❌ **NOT** comparing different images

**In one sentence**: For each image, we apply 4 different filtrations, compute 4 persistence diagrams, and get 4 Wasserstein amplitudes (one per filtration), all measuring distance to the empty diagram.

