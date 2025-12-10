'''
Principal Component Analysis (PCA) utilities for 3D geometry.

Provides eigen-decomposition of local point neighborhoods for computing:
- Surface normals
- Curvature metrics
- Local geometric structure
'''

import numpy as np

def pca(points: np.ndarray):
    '''
    Compute PCA on a set of neighboring 3D points.

    Parameters:
        points (np.ndarray):
            Nx3 array of points describing the neighborhood.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            eigenvalues, eigenvectors, mean
                eigenvalues:  (3,) sorted ascending
                eigenvectors: (3,3) columns are normalized eigenvectors
                mean:         (3,) centroid of neighborhood
    '''
    if points.shape[0] < 3:
        raise ValueError("PCA requires at least 3 points for a valid covariance matrix.")

    # Mean
    mean = np.mean(points, axis=0)
    centered = points - mean

    # Covariance matrix
    cov = np.dot(centered.T, centered) / points.shape[0]

    # Eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ensure sorted (ascending → smallest = surface normal dir)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors, mean
