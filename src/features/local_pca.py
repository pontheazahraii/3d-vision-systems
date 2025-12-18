'''
Local PCA computation for point cloud features.

This module provides a reusable abstraction for computing Principal
Component Analysis (PCA) on local neighborhoods of a point cloud.
It serves as a shared dependency for higher-level feature extractors
such as surface normals, curvature, and surface-type classification.

Neighborhoods are obtained via a spatial index (KDTree), and PCA
is performed on the resulting local point set.
'''

import numpy as np
from core.math.pca import pca
from core.spatial.kdtree import KDTree


class LocalPCA:
    '''
    Computes local PCA for points in a point cloud.

    This class encapsulates neighborhood querying and PCA computation,
    but does not interpret the PCA results. Feature extractors are
    responsible for interpreting eigenvalues and eigenvectors.
    '''

    def __init__(self, kdtree: KDTree, k_neighbors: int = 20):
        '''
        Initialize the LocalPCA helper.

        Args:
            kdtree (KDTree):
                Spatial index over the point cloud.
            k_neighbors (int):
                Number of nearest neighbors to use for local PCA computation.
        '''
        self.tree = kdtree
        self.k_neighbors = k_neighbors

    def compute(self, point):
        '''
        Compute PCA on the local neighborhood of a point.

        The neighborhood is obtained using k-nearest neighbors from
        the KDTree. PCA is then applied to the neighboring points to
        extract local geometric structure.

        Args:
            point (np.ndarray):
                3D query point of shape (3,).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - eigenvalues (3,): Sorted eigenvalues in ascending order.
                - eigenvectors (3, 3): Corresponding eigenvectors as columns.
                - mean (3,): Mean of the neighborhood points.
        '''
        neighbors = self.tree.k_nearest(point, self.k_neighbors)
        neighbor_points = np.array([node.point for (_, node) in neighbors])

        return pca(neighbor_points)
