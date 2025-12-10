'''
Surface normal and curvature estimation for point clouds.

Uses PCA on local neighborhoods queried from a KD-Tree.
'''

import numpy as np
from core.math.pca import pca
from core.spatial.kdtree import KDTree


class NormalEstimator:
    def __init__(self, kdtree: KDTree, k_neighbors: int = 20):
        '''
        Estimate surface normals and curvature using PCA.

        Parameters:
            kdtree (KDTree): Point spatial index.
            k_neighbors (int): Number of neighbors for PCA.
        '''
        self.tree = kdtree
        self.k_neighbors = k_neighbors

    def estimate_normals(self, points):
        '''Compute normals for a list of points.'''
        return [self._estimate_normal(p) for p in points]

    def estimate_curvatures(self, points):
        '''Compute curvature values for a list of points.'''
        return [self._estimate_curvature(p) for p in points]

    def estimate_all(self, points):
        '''Compute normals + curvature for all points.'''
        normals, curvatures = [], []
        for p in points:
            normal, curv = self._estimate_normal_and_curvature(p)
            normals.append(normal)
            curvatures.append(curv)
        return normals, curvatures

    def _estimate_normal(self, point):
        normal, _ = self._estimate_normal_and_curvature(point)
        return normal

    def _estimate_curvature(self, point):
        _, curvature = self._estimate_normal_and_curvature(point)
        return curvature

    def _estimate_normal_and_curvature(self, point):
        '''
        Runs PCA once and returns both the normal and curvature.
        '''
        neighbors = self.tree.k_nearest(point, self.k_neighbors)
        neighbor_points = np.array([node.point for (_, node) in neighbors])

        eigenvalues, eigenvectors, mean = pca(neighbor_points)

        # Normal = eigenvector of smallest eigenvalue
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)

        # Flip normal toward point for consistency
        view_dir = point - mean
        if np.dot(normal, view_dir) > 0:
            normal = -normal

        # Curvature: smallest eigenvalue / trace
        curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-12)

        return normal, curvature


if __name__ == '__main__':
    # Minimal usage test
    points = [(3, 6, 1), (17, 15, 2), (13, 15, 3), (5, 17, 6)]
    tree = KDTree(points)
    ne = NormalEstimator(tree, k_neighbors=2)

    normals, curvatures = ne.estimate_all(points)

    for p, n, c in zip(points, normals, curvatures):
        print(f'Point: {p}, Normal: {n}, Curvature: {c:.4f}')
