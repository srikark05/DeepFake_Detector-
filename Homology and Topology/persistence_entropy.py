"""
Custom implementation of Persistence Entropy
Computes the entropy of persistence diagrams
"""

import numpy as np


class PersistenceEntropy:
    """
    Compute persistence entropy from persistence diagrams.
    
    Persistence entropy is defined as:
    E = -sum(p_i * log(p_i))
    where p_i = (d_i - b_i) / sum(d_j - b_j)
    and (b_i, d_i) are the birth and death times of features.
    """
    
    def __init__(self):
        """Initialize the PersistenceEntropy transformer"""
        pass
    
    def fit(self, X, y=None):
        """Fit method for sklearn compatibility (no-op)"""
        return self
    
    def fit_transform(self, diagrams):
        """
        Compute persistence entropy for each diagram.
        
        Args:
            diagrams: Array of persistence diagrams
                      Shape: (n_samples, n_points, 3) where last dim is [dim, birth, death]
        
        Returns:
            entropies: Array of entropy values, shape (n_samples, n_homology_dims)
        """
        if isinstance(diagrams, list):
            diagrams = np.array(diagrams)
        
        # Handle single diagram
        if diagrams.ndim == 2:
            diagrams = diagrams[np.newaxis, :, :]
        
        n_samples = diagrams.shape[0]
        entropies_list = []
        
        for i in range(n_samples):
            diagram = diagrams[i]
            
            # Get unique homology dimensions
            dims = np.unique(diagram[:, 0].astype(int))
            sample_entropies = []
            
            for dim in dims:
                # Filter features for this dimension
                dim_features = diagram[diagram[:, 0] == dim]
                
                if len(dim_features) == 0:
                    sample_entropies.append(0.0)
                    continue
                
                # Extract birth and death times
                births = dim_features[:, 1]
                deaths = dim_features[:, 2]
                
                # Compute persistence (lifetime)
                persistences = deaths - births
                
                # Filter out zero or negative persistences
                persistences = persistences[persistences > 1e-10]
                
                if len(persistences) == 0:
                    sample_entropies.append(0.0)
                    continue
                
                # Normalize to get probabilities
                total_persistence = np.sum(persistences)
                if total_persistence < 1e-10:
                    sample_entropies.append(0.0)
                    continue
                
                probabilities = persistences / total_persistence
                
                # Compute entropy: -sum(p * log(p))
                # Use natural log and handle zeros
                log_probs = np.log(probabilities + 1e-10)
                entropy = -np.sum(probabilities * log_probs)
                
                sample_entropies.append(entropy)
            
            # Pad to ensure consistent shape (assume max 3 dimensions: 0, 1, 2)
            while len(sample_entropies) < 3:
                sample_entropies.append(0.0)
            
            entropies_list.append(sample_entropies[:3])  # Take first 3 dimensions
        
        result = np.array(entropies_list)
        
        # If single sample, return flattened
        if n_samples == 1:
            return result.flatten()
        
        return result

