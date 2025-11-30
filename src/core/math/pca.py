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
        normals = []
        for p in points:
            normals.append(self._estimate_normal(p))
        return normals

    def _estimate_normal(self, point):
        '''Estimates the normal at a single point using PCA on neighbors'''

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

        # 6. Normal = eigenvector of the smallest eigenvalue
        idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, idx]

        # 7. Normalize
        normal = normal / np.linalg.norm(normal)
        return normal


if __name__ == '__main__':
    points = [(3, 6, 1), (17, 15, 2), (13, 15, 3), (5, 17, 6)]
    tree = KDTree(points)

    ne = NormalEstimator(tree, k_neighbors=2)
    normals = ne.estimate_normals(points)

    for p, n in zip(points, normals):
        print(f'Point: {p}, Normal: {n}')