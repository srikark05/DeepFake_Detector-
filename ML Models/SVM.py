from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class TDASVMClassifier:
    def __init__(self, n_components=50, C=3, gamma="scale"):
        """
        n_components: PCA dimension (will be adjusted to min(n_components, n_features))
        C, gamma: SVM hyperparameters
        """
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.pca = None  # Will be initialized after seeing data

        # SVM with RBF kernel
        self.svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True
        )

        self.is_fit = False

    def fit(self, features, labels, test_size=0.2):
        """
        Train the TDA-SVM model.
        """
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=42, shuffle=True
        )

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize PCA with adaptive components
        max_components = min(self.n_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
        if self.pca is None:
            self.pca = PCA(n_components=max_components)
        elif self.pca.n_components > max_components:
            self.pca = PCA(n_components=max_components)

        # PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        # Train SVM
        self.svm.fit(X_train_pca, y_train)
        self.is_fit = True

        # Evaluate
        preds = self.svm.predict(X_test_pca)
        acc = accuracy_score(y_test, preds)

        print(f"\nSVM (RBF) Accuracy: {acc:.4f}\n")
        print(classification_report(y_test, preds))

        return acc, preds, y_test

    def transform_features(self, features):
        """
        Apply scaler + PCA to new features.
        Used by predict() and predict_proba().
        """
        if not self.is_fit:
            raise ValueError("Model must be trained before calling transform_features()")

        scaled = self.scaler.transform(features)
        reduced = self.pca.transform(scaled)
        return reduced

    def predict(self, features):
        """
        Predict class label (0 or 1) for new examples.
        """
        if not self.is_fit:
            raise ValueError("Model must be trained before calling predict()")

        features_transformed = self.transform_features(features)
        return self.svm.predict(features_transformed)

    def predict_proba(self, features):
        """
        Predict class probabilities for new examples.
        """
        if not self.is_fit:
            raise ValueError("Model must be trained before calling predict_proba()")

        features_transformed = self.transform_features(features)
        return self.svm.predict_proba(features_transformed)

