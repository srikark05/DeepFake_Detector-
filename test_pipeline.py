#!/usr/bin/env python3
"""
Quick test script to verify pipeline components work correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent / "ML Models"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from Filtration import Filtration
        print("✓ Filtration imported")
    except Exception as e:
        print(f"✗ Filtration import failed: {e}")
        return False
    
    try:
        from T_Diagrams import Persistence
        print("✓ Persistence imported")
    except Exception as e:
        print(f"✗ Persistence import failed: {e}")
        return False
    
    try:
        from Entropy import TDA_FeatureExtractor
        print("✓ TDA_FeatureExtractor imported")
    except Exception as e:
        print(f"✗ TDA_FeatureExtractor import failed: {e}")
        return False
    
    try:
        from SVM import TDASVMClassifier
        print("✓ TDASVMClassifier imported")
    except Exception as e:
        print(f"✗ TDASVMClassifier import failed: {e}")
        return False
    
    return True

def test_filtration():
    """Test Filtration class"""
    print("\nTesting Filtration class...")
    try:
        from Filtration import Filtration
        
        # Create sample points
        points = np.random.rand(100, 3)
        filt = Filtration(points)
        
        # Test Height
        direction = np.array([0, 0, 1])
        height_vals = filt.Height(direction)
        print(f"✓ Height filtration: {len(height_vals)} values")
        
        # Test Radial
        center = np.mean(points, axis=0)
        radial_vals = filt.Radial(center)
        print(f"✓ Radial filtration: {len(radial_vals)} values")
        
        # Test Density
        density_vals = filt.Density(k=5)
        print(f"✓ Density filtration: {len(density_vals)} values")
        
        # Test Dilation
        dilation_matrix = filt.Dilation()
        print(f"✓ Dilation filtration: {dilation_matrix.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Filtration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persistence():
    """Test Persistence class"""
    print("\nTesting Persistence class...")
    try:
        from T_Diagrams import Persistence
        
        # Create sample image
        image = np.random.rand(64, 64)
        persistence = Persistence(image)
        
        # Test diagram computation
        diagrams = persistence.compute_diagram()
        print(f"✓ Persistence diagrams computed: {len(diagrams)} diagrams")
        
        # Test feature extraction
        landscapes = persistence.compute_landscape(diagrams)
        print(f"✓ Landscapes computed: {landscapes.shape}")
        
        amplitudes = persistence.compute_amplitude(diagrams)
        print(f"✓ Amplitudes computed: {amplitudes.shape}")
        
        betti = persistence.compute_betti(diagrams)
        print(f"✓ Betti curves computed: {betti.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        data_dir = Path("preprocessing_raw_image_data")
        if not data_dir.exists():
            print(f"⚠ Data directory {data_dir} does not exist")
            return False
        
        test_gray = data_dir / "test_gray"
        if not test_gray.exists():
            print(f"⚠ Test gray directory does not exist")
            return False
        
        fake_dir = test_gray / "fake"
        real_dir = test_gray / "real"
        
        fake_count = len(list(fake_dir.glob("*.png"))) + len(list(fake_dir.glob("*.jpg")))
        real_count = len(list(real_dir.glob("*.png"))) + len(list(real_dir.glob("*.jpg")))
        
        print(f"✓ Found {fake_count} fake images")
        print(f"✓ Found {real_count} real images")
        
        return fake_count > 0 and real_count > 0
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Pipeline Component Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Filtration", test_filtration),
        ("Persistence", test_persistence),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! Pipeline should work correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


