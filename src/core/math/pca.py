'''
A method used to estimate the surface normal vector for each point in a 3D point cloud by analyzing the local neighborhood of each point.
'''

import numpy as np

from .kdtree import KDTree

class NormalEstimator:
    def __init__(self, kdtree, k_neighbors=20):
        '''
        Estimate surface normals using PCA.

        Parameters:
            kdtree (KDTree):
                The KDTree containing the points.
            k_neighbors (int):
                Number of neighbors to use for normal estimation.
        '''
        self.tree = kdtree
        self.k_neighbors = k_neighbors

    def estimate_normals(self, points):
        '''
        Compute PCA normals for each point

        Parameters:
            points (List[Tuple[float]]):
                Points to estimate normals for.

        Returns:
            List[np.ndarray]:
                A list of normal vectors (unit length).
        '''
        return [self._estimate_normal(p) for p in points]

    def estimate_curvatures(self, points):
        '''
        Compute curvature for each points.

        Parameters:
            points (List(Tuple[float])):
                Points to estimate curvature for.

        Returns:
            List[float]:
                Curvature values.
        '''
        return [self._estimate_curvature(p) for p in points]

    def estimate_all(self, points):
        '''
        Estimate both normals and curvature for each point.

        Parameters:
            points (List(Tuple[float])):
                Points to estimate normals and curvature for.

        Returns:
            Tuple[List[np.ndarray], List[float]]:
                (normals, curvatures)
        '''
        normals = []
        curvatures = []

        for p in points:
            normal, curvature = self._estimate_normal_and_curvature(p)
            normals.append(normal)
            curvatures.append(curvature)

        return normals, curvatures

    def _pca(self, point):
        '''Runs PCA on the KNN neighborhood of a point.'''
        # 1. Get k nearest neighbors from the KDTree
        neighbors = self.tree.k_nearest(point, self.k_neighbors)

        # Extract points only
        neighbor_points = np.array([node.point for _, node in neighbors])

        # 2. Compute the mean
        mean = np.mean(neighbor_points, axis=0)

        # 3. Subtract mean
        centered = neighbor_points - mean

        # 4. Covariance matrix
        cov = np.dot(centered.T, centered) / len(neighbor_points)

        # 5. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvalues, eigenvectors

    def _estimate_normal(self, point):
        '''Estimates the normal at a single point using PCA on neighbors'''
        eigenvalues, eigenvectors = self._pca(point)

        # Normal = eigenvector of the smallest eigenvalue
        idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, idx]

        # Normalize
        normal = normal / np.linalg.norm(normal)
        return normal

    def _estimate_curvature(self, point):
        '''
        Estimate curvature using PCA eigenvalues.

        curvature = λ_min / (λ0 + λ1 + λ2)
        '''
        eigenvalues, _ = self._pca(point)
        l0, l1, l2 = eigenvalues
        curvature = l0 / (l0 + l1, + l2)
        return curvature

    def _estimate_normal_and_curvature(self, point):
        '''Single PCA call that returns both normal and curvature'''
        eigenvalues, eigenvectors = self._pca(point)

        # Normal
        idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, idx]
        normal = normal / np.linalg.norm(normal)

        # Curvature
        l0, l1, l2 = eigenvalues
        curvature = l0 / (l0 + l1 + l2)

        return normal, curvature


if __name__ == '__main__':
    points = [(3, 6, 1), (17, 15, 2), (13, 15, 3), (5, 17, 6)]
    tree = KDTree(points)

    ne = NormalEstimator(tree, k_neighbors=2)
    normals, curvature = ne.estimate_all(points)

    for p, n, c in zip(points, normals, curvature):
        print(f'Point: {p}, Normal: {n}, Curvature: {c}')