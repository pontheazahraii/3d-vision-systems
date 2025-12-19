'''
Surface type classification based on local PCA-derived metrics.

This module classifies points into coarse geometric surface types
(e.g. planar, linear, scattered) using interpretable thresholds
applied to suface metrics such as planarity, linearity, and curvature.

The classifier operates on a per-point basis and does not perform
segmentation or region growing.
'''

from enum import Enum
from perception_systems.features.surface_metrics import SurfaceMetrics

class SurfaceType(Enum):
    '''
    Enumerates supported surface types.
    '''
    PLANAR = 'planar'
    LINEAR = 'linear'
    SCATTERED = 'scattered'
    UNDEFINED = 'undefined'


class SurfaceClassifier:
    '''
    Classifies local suface structure using PCA-based surface metrics.

    This class applies threshold-based rules to surface metrics in order
    to assign each point a surface type. Thersholds are intentionally
    explicity and interpretable to allow debugging and tuning.
    '''
    def __init__(
            self,
            surface_metrics: SurfaceMetrics,
            planarity_thresh: float = 0.5,
            linearity_thresh: float = 0.5,
            curvature_thresh: float = 0.05,
    ):
        '''
        Initialize the surface classifier.

        Args:
            surface_metrics (SurfaceMetrics):
                Surface metrics computation object.
            planarity_thresh (float):
                Minimum planarity value to classify a point as planar.
            linearity_thresh (float):
                Minimum linearity values to classify a point as linear.
            curvature_thresh (float):
                Maximum curvature for planar or linear classification.
        '''
        self.metrics = surface_metrics
        self.planarity_thresh = planarity_thresh
        self.linearity_thresh = linearity_thresh
        self.curvature_thresh = curvature_thresh

    def classify(self, point):
        '''
        Classify the surface type at a single point.

        Args:
            point (np.ndarray):
                3D query point of shape (3,).
        
        Returns:
            SurfaceType:
                Classified surface type.
        '''
        m = self.metrics.compute(point)

        curvature = m['curvature']
        planarity = m['planarity']
        linearity = m['linearity']

        # Reject high-curvature points early
        if curvature > self.curvature_thresh: return SurfaceType.SCATTERED
        
        # Planar surfaces 
        if planarity >= self.planarity_thresh: return SurfaceType.PLANAR

        # Linear structures (edges, corners)
        if linearity >= self.linearity_thresh: return SurfaceType.LINEAR

        return SurfaceType.UNDEFINED
    
    def classify_batch(self, points):
        '''
        Classify surface types for multiple points.

        Args:
            points (Iterable[np.ndarray]):
                Iterable of 3D points.

        Returns:
            List[SurfaceType]:
                List of surface classifications.
        '''
        return [self.classify(p) for p in points]
    

if __name__ == "__main__":
    import numpy as np
    from perception_systems.core.spatial.kdtree import KDTree
    from perception_systems.features.local_pca import LocalPCA

    # Synthetic mixed-structure cloud
    plane = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    line = np.array([
        [2.0, 0.0, 0.0],
        [2.0, 0.5, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 1.5, 0.0],
    ])

    noise = np.array([
        [3.1, 0.3, 0.7],
        [3.4, 1.2, 0.4],
        [3.2, 0.8, 1.1],
    ])

    points = np.vstack([plane, line, noise])

    tree = KDTree(points.tolist())
    local_pca = LocalPCA(tree, k_neighbors=5)
    metrics = SurfaceMetrics(local_pca)
    classifier = SurfaceClassifier(metrics)

    print("Surface classification test:")
    for p in points:
        label = classifier.classify(p)
        print(f"Point {p} → {label.value}")