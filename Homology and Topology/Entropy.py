import numpy as np
from persistence_entropy import PersistenceEntropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TDA_FeatureExtractor:
    def __init__(self):
        self.entropy = PersistenceEntropy()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)

    def extract_features(self, diagrams, persistence_module):
        """
        diagrams: output of Persistence.compute_diagram()
        persistence_module: instance of your Persistence class
        """

        # 1. Persistence landscapes
        L = persistence_module.compute_landscape(diagrams)
        if L.ndim > 1:
            L = L.reshape(L.shape[0], -1)   # flatten
        else:
            L = L.reshape(1, -1)

        # 2. Entropy
        E = self.entropy.fit_transform(diagrams)
        if E.ndim == 1:
            E = E.reshape(1, -1)
        elif E.ndim == 0:
            E = np.array([[E]])

        # 3. Wasserstein amplitude
        A = persistence_module.compute_amplitude(diagrams)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        elif A.ndim == 0:
            A = np.array([[A]])

        # 4. Betti curves
        B = persistence_module.compute_betti(diagrams)
        if B.ndim > 1:
            B = B.reshape(B.shape[0], -1)
        else:
            B = B.reshape(1, -1)

        # Ensure all have same first dimension (batch size)
        batch_size = L.shape[0]
        feature_list = []
        
        if L.shape[0] == batch_size:
            feature_list.append(L)
        if E.shape[0] == batch_size:
            feature_list.append(E)
        if A.shape[0] == batch_size:
            feature_list.append(A)
        if B.shape[0] == batch_size:
            feature_list.append(B)

        if len(feature_list) == 0:
            return np.zeros((batch_size, 1))
        
        features = np.concatenate(feature_list, axis=1)
        return features

    def fit_pca(self, features):
        scaled = self.scaler.fit_transform(features)
        reduced = self.pca.fit_transform(scaled)
        return reduced

    def transform_pca(self, features):
        scaled = self.scaler.transform(features)
        reduced = self.pca.transform(scaled)
        return reduced
