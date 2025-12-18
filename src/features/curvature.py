'''
Surface curvature estimation for point clouds.

This module computes a scalar curvature measure for each point based
on eigenvalue ratios obtained from local PCA. Curvature captures the
degree of surface variation and is useful for surface classification,
segmentation, and feature detection.
'''

from features.local_pca import LocalPCA


class CurvatureEstimator:
    '''
    Estimates surface curvature using local PCA eigenvalues.

    Curvature is computed as the ratio of the smallest eigenvalue to
    the sum of all eigenvalues of the local covariance matrix:

        curvature = λ_min / (λ0 + λ1 + λ2)

    Lower values indicate planar regions, while higher values indicate
    edges, corners, or noisy structures.
    '''

    def __init__(self, local_pca: LocalPCA):
        '''
        Initialize the curvature estimator.

        Args:
            local_pca (LocalPCA):
                Shared local PCA computation object.
        '''
        self.local_pca = local_pca

    def estimate(self, point):
        '''
        Estimate surface curvature at a single point.

        Args:
            point (np.ndarray):
                3D point of shape (3,).

        Returns:
            float:
                Scalar curvature value.
        '''
        eigenvalues, _, _ = self.local_pca.compute(point)
        return eigenvalues[0] / (eigenvalues.sum() + 1e-12)

    def estimate_batch(self, points):
        '''
        Estimate surface curvature for multiple points.

        Args:
            points (Iterable[np.ndarray]):
                Iterable of 3D points.

        Returns:
            List[float]:
                List of curvature values.
        '''
        return [self.estimate(p) for p in points]
    

if __name__ == '__main__':
    import numpy as np
    from core.spatial.kdtree import KDTree
    from features.local_pca import LocalPCA

    # Same planar point set for curvature sanity check
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    tree = KDTree(points)
    local_pca = LocalPCA(tree, k_neighbors=4)
    curvature_estimator = CurvatureEstimator(local_pca)

    print('Curvature estimation test:')
    for p in points:
        c = curvature_estimator.estimate(p)
        print(f'Point {p} → Curvature {c:.6f}')