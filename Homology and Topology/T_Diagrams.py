import numpy as np
from gtda.homology import CubicalPersistence 
from gtda.diagrams import PersistenceLandscape, Amplitude, BettiCurve
from gtda.plotting import plot_diagram


class Persistence:
    def __init__(self, images):
        """
        images:
            - Single 2D array (H, W)
            - OR batch of 2D arrays (N, H, W)
        """
        self.images = images

        # Cubical persistence (metric removed — not valid here)
        self.cube = CubicalPersistence(
            homology_dimensions=[0, 1, 2]
        )

        # Diagram vectorizers
        self.landscape = PersistenceLandscape(n_layers=5)
        self.amplitude = Amplitude(metric="wasserstein")
        self.betti = BettiCurve()


    def compute_diagram(self):
        """
        Computes cubical persistence diagram(s).
        Handles both single images and batches.
        Returns: diagrams, shape (N, n_points, 3)
        """
        # Batch handling:
        if self.images.ndim == 2:
            # Single image → add batch dimension
            imgs = self.images[None, :, :]
        else:
            imgs = self.images

        diagrams = self.cube.fit_transform(imgs)
        return diagrams


    def plot_diagram(self, diagrams):
        """
        Plot diagram of first sample in batch.
        """
        return plot_diagram(diagrams[0])


    def compute_landscape(self, diagrams):
        """
        Returns persistence landscapes.
        Shape: (N, layers * samples_per_layer)
        """
        return self.landscape.fit_transform(diagrams)


    def compute_amplitude(self, diagrams):
        """
        Wasserstein amplitude (scalar or vector).
        """
        return self.amplitude.fit_transform(diagrams)


    def compute_betti(self, diagrams):
        """
        Betti curves.
        """
        return self.betti.fit_transform(diagrams)
