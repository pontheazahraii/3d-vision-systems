'''
Surface normal estimation for point clouds.

This module estimates per-point surface normals using PCA on local
neighborhoods queried from a spatial index. The normal direction is
defined as the eigenvector corresponding to the smallest eigenvalue
of the local covariance matrix.

Normal orientation is made consistent by flipping the normal to
face toward the query point relative to the neighborhood mean.
'''

import numpy as np
from perception_systems.features.local_pca import LocalPCA


class NormalEstimator:
    '''
    Estimates surface normals for point clouds using local PCA.

    This class interprets local PCA results as surface orientation.
    It does not perform neighborhood search or PCA directly; instead,
    it relies on a LocalPCA object for those computations.
    '''

    def __init__(self, local_pca: LocalPCA):
        '''
        Initialize the normal estimator.

        Args:
            local_pca (LocalPCA):
                Shared local PCA computation object.
        '''
        self.local_pca = local_pca

    def estimate(self, point):
        '''
        Estimate the surface normal at a single point.

        The normal is defined as the eigenvector associated with the
        smallest eigenvalue of the local covariance matrix. The normal
        is flipped to ensure consistent orientation relative to the
        neighborhood mean.

        Args:
            point (np.ndarray):
                3D point of shape (3,).

        Returns:
            np.ndarray:
                Unit surface normal vector of shape (3,).
        '''
        _, eigenvectors, mean = self.local_pca.compute(point)

        # Eigenvector of smallest eigenvalue
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)

        # Flip normal for consistent orientation
        view_dir = point - mean
        if np.dot(normal, view_dir) > 0:
            normal = -normal

        return normal

    def estimate_batch(self, points):
        '''
        Estimate surface normals for multiple points.

        Args:
            points (Iterable[np.ndarray]):
                Iterable of 3D points.

        Returns:
            List[np.ndarray]:
                List of unit surface normal vectors.
        '''
        return [self.estimate(p) for p in points]


if __name__ == '__main__':
    import numpy as np
    from perception_systems.core.spatial.kdtree import KDTree
    from perception_systems.features.local_pca import LocalPCA

    # Simple synthetic point set (rough plane)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    tree = KDTree(points)
    local_pca = LocalPCA(tree, k_neighbors=4)
    normal_estimator = NormalEstimator(local_pca)

    print('Normal estimation test:')
    for p in points:
        n = normal_estimator.estimate(p)
        print(f'Point {p} → Normal {n}')