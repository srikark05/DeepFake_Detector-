# Wasserstein Distance - Visual Explanation

## The Answer: Same Image, Different Filtrations

**Wasserstein amplitude is computed on the SAME image, but for EACH filtration separately.**

## Visual Flow for ONE Image

```
┌─────────────────────────────────────────────────────────┐
│                    ONE IMAGE                            │
│              (e.g., Image_001.jpg)                      │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐
   │ Height │      │ Radial │      │Density │      │Dilation│
   │  Filt  │      │  Filt  │      │  Filt  │      │  Filt  │
   └────────┘      └────────┘      └────────┘      └────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
   │Filtered│      │Filtered│      │Filtered│      │Filtered│
   │ Image  │      │ Image  │      │ Image  │      │ Image  │
   │(height)│      │(radial)│      │(density)│     │(dilation)│
   └────────┘      └────────┘      └────────┘      └────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
   │Diagram │      │Diagram │      │Diagram │      │Diagram │
   │   D₁   │      │   D₂   │      │   D₃   │      │   D₄   │
   └────────┘      └────────┘      └────────┘      └────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
   │W(D₁,Δ) │      │W(D₂,Δ) │      │W(D₃,Δ) │      │W(D₄,Δ) │
   │ = 0.45 │      │ = 0.32 │      │ = 0.67 │      │ = 0.28 │
   └────────┘      └────────┘      └────────┘      └────────┘
        │               │               │               │
        └───────────────┼───────────────┘
                        ▼
            ┌───────────────────────┐
            │  Feature Vector for   │
            │     Image_001         │
            │                       │
            │ [..., 0.45, 0.32,     │
            │      0.67, 0.28, ...] │
            └───────────────────────┘
```

## Key Points

### ✅ What We DO:

1. **Same Image**: All filtrations start from the same original image
2. **Different Filtrations**: Each filtration creates a different filtered view
3. **Independent Computation**: Each filtration gets its own:
   - Persistence diagram
   - Wasserstein amplitude (distance to empty diagram)

### ❌ What We DON'T Do:

1. **NOT comparing filtrations**: We don't compute W(D₁, D₂)
2. **NOT comparing images**: We don't compute W(Image_A, Image_B)
3. **NOT cross-filtration**: We don't mix diagrams from different filtrations

## Code Flow (from pipeline.py)

```python
# For ONE image:
for filt_name, grid in filtered_grids.items():  # Loop through 4 filtrations
    # grid = filtered version of the SAME image
    diagrams = compute_diagram(grid)  # One diagram per filtration
    amplitude = compute_amplitude(diagrams)  # W(diagram, empty)
    # This gives us 4 amplitudes for the same image
```

## Example: Image_001

**Input**: Image_001 (one image file)

**Process**:
1. Apply Height Filtration → Height-filtered Image_001
   - Compute Diagram D₁
   - Compute W(D₁, empty) = 0.45

2. Apply Radial Filtration → Radial-filtered Image_001  
   - Compute Diagram D₂
   - Compute W(D₂, empty) = 0.32

3. Apply Density Filtration → Density-filtered Image_001
   - Compute Diagram D₃
   - Compute W(D₃, empty) = 0.67

4. Apply Dilation Filtration → Dilation-filtered Image_001
   - Compute Diagram D₄
   - Compute W(D₄, empty) = 0.28

**Output**: 4 Wasserstein amplitudes for Image_001

## Why This Makes Sense

Each filtration reveals **different topological structures** in the same image:
- **Height**: Shows intensity-based topology
- **Radial**: Shows distance-based topology  
- **Density**: Shows local density topology
- **Dilation**: Shows distance-relationship topology

By computing amplitude for each, we capture **multiple perspectives** of the same image's topology.

## Summary Table

| Aspect | Answer |
|--------|--------|
| **Same or Different Images?** | ✅ **Same image** |
| **Same or Different Filtrations?** | ✅ **Different filtrations** (4 types) |
| **What is Compared?** | Each diagram vs **empty diagram** (Δ) |
| **How Many Amplitudes per Image?** | **4** (one per filtration) |
| **Are Filtrations Compared?** | ❌ No - each is independent |

## Final Answer

**Wasserstein distance is computed on the SAME image, but for EACH of the 4 different filtrations separately.**

Each filtration of the same image produces:
- Its own filtered version
- Its own persistence diagram  
- Its own Wasserstein amplitude (distance to empty diagram)

This gives us 4 independent topological measurements of the same image, each from a different perspective.

