#!/usr/bin/env python3
"""
Train and save the TDA-based DeepFake Detection model
Run this script first to train the model before using the API
"""

import sys
import pickle
from pathlib import Path
import numpy as np

# Add module paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Filtration"))
sys.path.insert(0, str(Path(__file__).parent / "Homology and Topology"))
sys.path.insert(0, str(Path(__file__).parent / "ML Models"))
sys.path.insert(0, str(Path(__file__).parent / "Preprocessing"))

from pipeline import main

def train_and_save_model():
    """Train the model using the pipeline and save it"""
    print("=" * 60)
    print("Training DeepFake Detection Model")
    print("=" * 60)
    
    # Run the main pipeline to train the model
    classifier, features, labels = main()
    
    if classifier is None:
        print("Error: Model training failed")
        return
    
    # Save the model
    model_path = Path(__file__).parent / "trained_model.pkl"
    target_size = (128, 128)  # Match the size used in pipeline
    
    model_data = {
        'classifier': classifier,
        'target_size': target_size,
        'feature_shape': features.shape if features is not None else None
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to {model_path}")
    print("You can now start the API server with: python app.py")
    print("Or run: uvicorn app:app --reload")

if __name__ == "__main__":
    try:
        train_and_save_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError in model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

