'''
Region growing segmentation for point clouds.

This module implements a basic region growing algorithm that builds
clusters by expanding from seed points into neighboring points that
satisfy geometry consistency constraints.

Common constraints:
    - Normal smoothness: neighbor normals must be within an angle threshold.
    - Curvature threshold: exlcude points with high curvature
    - Optional surface-type gating: grow only specific sufrance types
    (e.g. planar-only)
'''

from collections import deque
from typing import Iterable, List, Optional, Tuple

import numpy as np

from perception_systems.core.spatial.kdtree import KDTree
from perception_systems.features.normals import NormalEstimator
from perception_systems.features.curvature import CurvatureEstimator
from perception_systems.features.surface_classifier import SurfaceClassifier, SurfaceType


class RegionGrowingSegmenter:
    '''
    Region growing segmenter from point clouds.

    This algorithm builds clusters by starting from seed point and 
    growing into neighbors that pass:
        - distance constraints (radius or KNN neighborhood)
        - normal angle consistency
        - curvature constraint
        - optional surface-type constraints
    
    Notes:
        - For best results, use radius neighborhood for organized surfaces.
        - kNN is often easier to tune for varying density.
    '''
    def __init__(
            self,
            kdtree: KDTree,
            normal_estimator: NormalEstimator,
            curvature_estimator: CurvatureEstimator,
            *,
            use_radius: bool = True,
            radius: float,
            k_neighbors: int = 30,
            smoothness_angle_deg: float = 15.0,
            curvature_thresh: float = 0.05,
            min_cluster_size: int = 50,
            max_cluster_size: int = 200000,
            surface_classifier: Optional['SurfaceClassifier'] = None,
            allowed_surface_types: Optional[Iterable['SurfaceType']] = None,
    ):
        '''
        Initialize region growing.

        Args:
            kdtree (KDTree):
                Spatial index over the point cloud points.
            normal_estimator (NormalEstimator):
                Estimates normals for points.
            curvature_estimator (CurvatureEstimator):
                Estimates curvature for points.
            use_radius (bool):
                If True, use radius search; else use kNN.
            radius (float):
                Radius for neighbor search if use_radius=True.
            k_neighbors (int):
                k for kNN search if use_radius=False.
            smoothness_angle_deg (float):
                Maximum allowed angle between normals
                (in degrees) for points to be in the same region.
            curvature_thresh (float):
                Maximum allowed curvature for inclusion.
            min_cluster_size (int):
                Minimum number of points for a cluster.
            max_cluster_size (int): 
                Maximum number of points for a cluster.
            surface_classifier (SurfaceClassifier, optional):
                If provided, used to gate seeds and growth based on surface type.
            allowed_surface_types (Iterable[SurfaceType], optional): 
                Allowed types for seeds and growth. If None and surface_classifier
                is set, defaults to {SurfaceType.PLANAR}.
        '''
        self.tree = kdtree
        self.ne = normal_estimator
        self.ce = curvature_estimator

        self.use_radius = use_radius
        self.radius = radius
        self.k_neighbors = k_neighbors

        self.smoothness_angle_deg = smoothness_angle_deg
        self.curvature_thresh = curvature_thresh

        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        self.surface_classifier = surface_classifier
        if self.surface_classifier is not None:
            self.allowed_surface_types = {SurfaceType.PLANAR} if allowed_surface_types is None else set(allowed_surface_types)
        else:
            self.allowed_surface_types = None
        
        # Precomputed consine threshold once
        theta = np.deg2rad(self.smoothness_angle_deg)
        self.cos_smoothness = float(np.cos(theta))

    def segment(
      self,
      points: np.ndarray,
      *,
      normals: Optional[np.ndarray] = None,
      curvatures: Optional[np.ndarray] = None,      
    ):
        '''
        Performs region growing segmentation.

        Args:
            point (np.ndarray):
                Array of shape (N, 3).
            normals (np.ndarray, optional):
                Precomputed normals of shape (N, 3).
                If not provided, they will be estimated.
            curvatures (np.ndarray, optional):
                Precomputed curvatures of shape (N,).
                If not provided, they will be estimated.

        Returns:
            List[List[int]]:
                List of clusters, each a list of point indices.
        
        Raises:
            ValueError:
                - If `points` is not a NumPy array of shape (N, 3).
                - If provided `normals` does not have shape (N, 3).
                - If provided `curvatures` does not have shape (N,).
            RuntimeError:
                - If surface-type gating is enabled but no `SurfaceClassifier`
                is configured.
        '''
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points must be a numpy array of shape (N, 3)')
        
        n = points.shape[0]

        # Compute features if not provided 
        if normals is None:
            normals = np.array(self.ne.estimate_batch(points), dtype=float)
        if curvatures is None:
            curvatures = np.array(self.ce.estimate_batch(points), dtype=float)

        if normals.shape != (n, 3):
            raise ValueError('normals must have shape (N, 3)')
        if curvatures.shape != (n,):
            raise ValueError('curvatures must have shape (N,)')
        
        visited = np.zeros(n, dtype=bool)
        clusters: List[List[int]] = []

        # Helper: classify a point if gating is enabled
        def allowed(i: int):
            if self.surface_classifier is None: return True
            st = self.surface_classifier.classify(points[i])
            return st in self.allowed_surface_types
        
        # Helper: get neighbor indicies for a point
        def neighbor_indices(i: int):
            p = points[i]
            if self.use_radius:
                neighbors = self.tree.radius_search(tuple(p), self.radius)
            else:
                neighbors = self.tree.k_nearest(tuple(p), self.k_neighbors)
            return [point_to_index[tuple(node.point)] for _, node in neighbors]
        
        point_to_index = {tuple(points[i]): i for i in range(n)}

        for seed in range(n):
            if visited[seed]: continue

            # Seed gating
            if curvatures[seed] > self.curvature_thresh:
                visited[seed] = True
                continue
            if not allowed(seed):
                visited[seed] = True
                continue

            # Start a new cluster (BFS)
            cluster: List[int] = []
            q = deque([seed])
            visited[seed] = True

            while q:
                i = q.popleft()
                cluster.append(i)

                if len(cluster) >= self.max_cluster_size: break

                ni = normals[i]
                ni_norm = np.linalg.norm(ni) + 1e-12

                for j in neighbor_indices(i):
                    if visited[j]: continue

                    # Curvature constraints
                    if curvatures[j] > self.curvature_thresh:
                        visited[j] = True
                        continue

                    # Surface-type gating
                    if not allowed(j):
                        visited[j] = True
                        continue

                    # Normal smoothness constaint
                    nj = normals[j]
                    nj_norm = np.linalg.norm(nj) + 1e-12
                    cosang = float(np.dot(ni, nj) / (ni_norm * nj_norm))

                    if cosang >= self.cos_smoothness:
                        visited[j] = True
                        q.append(j)
            
            # Keep cluster only if it meets size constraints
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(cluster)

        return clusters

if __name__ == "__main__":
    # Minimal sanity test:
    # Two planar patches + scattered noise, region growing should form ≥ 2 clusters.

    from perception_systems.features.local_pca import LocalPCA

    np.random.seed(7)

    # Plane A: z=0 around (0,0)
    plane_a = np.hstack([
        np.random.uniform(-0.5, 0.5, (400, 2)),
        np.zeros((400, 1)),
    ])

    # Plane B: z=0 around (2,0)
    plane_b = np.hstack([
        np.random.uniform(1.5, 2.5, (400, 2)),
        np.zeros((400, 1)),
    ])

    # Noise: random 3D
    noise = np.random.uniform([-0.5, -0.5, -0.5], [2.5, 0.5, 0.5], (60, 3))

    points = np.vstack([plane_a, plane_b, noise])

    # Build KDTree expects list of tuples
    tree = KDTree(points.tolist())

    # LocalPCA uses KDTree + k-neighborhood internally
    local_pca = LocalPCA(tree, k_neighbors=20)
    ne = NormalEstimator(local_pca)
    ce = CurvatureEstimator(local_pca)

    segmenter = RegionGrowingSegmenter(
        tree,
        ne,
        ce,
        use_radius=True,
        radius=0.20,
        smoothness_angle_deg=20.0,
        curvature_thresh=0.03,
        min_cluster_size=50,
    )

    clusters = segmenter.segment(points)

    print(f"Found {len(clusters)} clusters")
    sizes = sorted([len(c) for c in clusters], reverse=True)
    print("Top cluster sizes:", sizes[:10])

    # Print a quick summary of cluster centroids for sanity
    for idx, c in enumerate(clusters[:5]):
        centroid = points[c].mean(axis=0)
        print(f"Cluster {idx}: size={len(c)}, centroid={centroid}")