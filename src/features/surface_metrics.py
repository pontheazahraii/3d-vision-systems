'''
Surface metrics derived from local PCA eigenvalues.

This module computes interpretable scalar metrics that describe the
local geometry structure around a point. These metrics are commonly
used for surface classification, region growing, and segmentation
control.

All metrics are computed solely from PCA eigenvalues and do not perform
neighborhood queries or PCA themselves.
'''

import numpy as np
from features.local_pca import LocalPCA


class SurfaceMetrics:
    '''
    Computes surface structure metrics from local PCA eigenvalues.

    The eigenvalues are assumed to be sorted in ascending order:
        λ0 ≤ λ1 ≤ λ2

    Metrics implemented:
        - curvature
        - linearity
        - planarity
        - scattering
        - anisotropy
    '''
    def __init__(self, local_pca: LocalPCA):
        '''
        Initialize the surface metrics calculator.

        Args:
            local_pca (LocalPCA):
                Shared local PCA computation object.
        '''
        self.local_pca = local_pca

    def compute(self, point):
        '''
        Compute all surface metrics for a single point.

        Args:
            point (np.ndarray):
                3D query point of shape (3,).
        
        Returns:
            dict:
                Dictionary containing surface metrics:
                    {
                        'curvature' : float,
                        'linearity' : float,
                        'planarity' : float,
                        'scattering': float,
                        'anisotropy': float
                    }
        '''
        eigenvalues, _ = self.local_pca.compute(point)
        l0, l1, l2 = eigenvalues + 1e-12  # numerical stability

        curvature = l0 / (l0 + l1 + l2)
        linearity = (l2 - l1) / l2
        planarity = (l1 - l0) / l2
        scattering = l0 / l2
        anisotropy = (l2 - l0) / l2

        return {
            'curvature' : curvature,
            'linearity' : linearity,
            'planarity' : planarity,
            'scattering': scattering,
            'anisotropy': anisotropy,
        }
    
    def compute_batch(self, points):
        '''
        Compute surface metrics for multiple points.

        Args:
            points (Iterable[np.ndarray]):
                Iterable of 3D points.

        Returns:
            List[dict]:
                List of metric dictionaries.
        '''
        return [self.compute(p) for p in points]


if __name__ == '__main__':
    from core.spatial.kdtree import KDTree

    # Synthetic planar cloud (z = 0)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    tree = KDTree(points)
    local_pca = LocalPCA(tree, k_neighbors=4)
    metrics = SurfaceMetrics(local_pca)

    print('Surface metrics test (planar cloud):')
    for p in points:
        m = metrics.compute(p)
        print(f'\nPoint {p}')
        for k, v in m.items():
            print(f'  {k:12s}: {v:.6f}')