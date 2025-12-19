'''
A KD-Tree (K-Dimensional Tree) is a space-partitioning data structure used 
to organize points in a k-dimensional space, facilitating efficient search 
operations like nearest neighbor queries
'''


class KDNode:
    '''
    Node representation for a KD-Tree.

    A KDNode stores a single point in k-dimensional space and references
    to its left and right child nodes. Each node implicitly represents
    a splitting hyperplane orthogonal to one coordinate axis, determined
    by the node's depth in the tree.

    Attributes:
        point (Tuple[float, ...]):
            The k-dimensional point stored at this node.
        k (int):
            The splitting axis index for this node.
        left (KDNode or None):
            Left child node containing points with smaller coordinate
            values along the splitting axis.
        right (KDNode or None):
            Right child node containing points with larger or equal
            coordinate values along the splitting axis.
    '''
    def __init__(self, point, k, left=None, right=None):
        '''
        Initialize the node for the KD-Tree.

        Parameters:
            point (Tuple[float, ...]):
                An array representing the coordinates in k-dimensional space.
                Must be indexable (e.g., tuple) and have length `k`.
            k (int):
                The dimensionality of the space the KD-Tree operates in.
                Defines how many coordinate axes the point has.
            left (KDNode, optional):
                The left child node. Contains points whose value on the splitting axis is less than the current node's point.
                Default is None.
            right (KDNode, optional):
                The right child node. Contains points whose value on the splitting axis is greater than or equal to the current node's point.
                Default is None.
        '''
        self.point = point
        self.k = k
        self.right = right
        self.left = left


class KDTree:
    '''
    K-Dimensional Tree for efficient spatial queries.

    This class implements a KD-Tree data structure for organizing points
    in k-dimensional space. It supports efficient nearest-neighbor,
    k-nearest-neighbor, and radius-based search queries.

    The tree is constructed recursively by alternating splitting axes
    and partitioning points around median values to maintain balance.

    Typical use cases include:
        - Nearest neighbor search
        - Local neighborhood queries for PCA
        - Spatial indexing for point cloud processing

    Attributes:
        k (int):
            Dimensionality of the space.
        root (KDNode):
            Root node of the KD-Tree.
    '''
    def __init__(self, points):
        '''
        Initializes and builds the KD-Tree.

        Parameters:
            points(List[Tuple[float, ...]]):
                A list of k-dimensional points.

        Raise:
            ValueError:
                If point list is empty.
        '''
        if points is None or len(points) == 0:
            raise ValueError("Points list cannot be empty.")

        self.k = len(points[0])
        self.root = self._build(points)

    def _build(self, points, depth=0):
        '''Recursively builds the KD-Tree.'''
        if points is None or len(points) == 0:
            return None

        points = list(points)
        axis = depth % self.k

        # Sort points along current axis
        points.sort(key=lambda p: p[axis])
        median_idx = len(points) // 2
        median_point = points[median_idx]

        node = KDNode(median_point, axis)
        node.left = self._build(points[:median_idx], depth+1)
        node.right = self._build(points[median_idx+1:], depth+1)

        return node

    @staticmethod
    def _distance_squared(p1, p2):
        '''Distance helper. Computes squared Euclidean distance between two k-d points.'''
        return sum((a-b) ** 2 for a, b in zip(p1, p2))

    def nearest(self, target):
        '''
        Finds the nearest neighbor to the target point.

        Parameters:
            target (Tuple[float, ...]):
                The query point.

        Returns:
            KDNode:
                Node containing the nearest point.
        '''
        best = []
        self._search(self.root, target, k=1, best=best, depth=0)
        return best[0][1]  # return KDNode only

    def k_nearest(self, target, k):
        '''
        Find the k nearest neighbors to the target point.

        Parameters:
            target(Tuple[float, ...]):
                The query point.
            k (int):
                Number of nearest neighbors to return.

        Returns:
            List[Tuple[float, KDNode]]:
                A list of (distance_squared, KDNode) tuples sorted by increasing distance.
        '''
        best = []  # list of (dist2, KDNode)
        self._search(self.root, target, k, best, depth=0)
        return sorted(best, key=lambda x: x[0])

    def _search(self, node, target, k, best, depth):
        '''Unified search for 1-NN and k-NN.'''
        if node is None:
            return

        axis = depth % self.k
        dist2 = self._distance_squared(target, node.point)

        # Insert/update best list
        if len(best) < k:
            best.append((dist2, node))
        else:
            # Replace worst (largest dist2) if this one is better
            worst_dist, worst_node = max(best, key=lambda x: x[0])
            if dist2 < worst_dist:
                best.remove((worst_dist, worst_node))
                best.append((dist2, node))

        # Choose direction
        if target[axis] < node.point[axis]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        # Search primary branch
        self._search(next_branch, target, k, best, depth+1)

        # Check splitting plane for potential other-branch search
        if len(best) < k:
            must_search_other = True
        else:
            plane_dist2 = (target[axis] - node.point[axis])**2
            worst_dist, _ = max(best, key=lambda x: x[0])
            must_search_other = plane_dist2 < worst_dist

        if must_search_other:
            self._search(other_branch, target, k, best, depth+1)

    def radius_search(self, target, radius):
        '''
        Find all neighbors within a given radius of a target point.

        Parameters:
            target (Tuple[float, ...]):
                The query point.
            radius (float):
                Maximum distance for neighbor inclusion.

        Returns:
            List[Tuple[float, KDNode]]:
                A list of (distance_squared, KDNode) tuples sorted by increasing distance.
        '''
        results = []
        radius2 = radius**2
        self._radius_search(self.root, target, radius2, results, depth=0)
        return sorted(results, key=lambda x: x[0])

    def _radius_search(self, node, target, radius2, results, depth):
        if node is None:
            return
        
        axis = depth % self.k
        dist2 = self._distance_squared(target, node.point)

        # Check if this node is within the search radius
        if dist2 <= radius2:
            results.append((dist2, node))
        
        # Decide traversal order 
        diff = target[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # Search the nearer side first 
        self._radius_search(near, target, radius2, results, depth+1)

        # Check whether the hypersphere intersects the splitting plane
        if diff**2 <= radius2:
            self._radius_search(far, target, radius2, results, depth+1)


if __name__ == '__main__':
    points = [(3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7)]
    tree = KDTree(points)

    target_point = (20, 17)
    nearest = tree.nearest(target_point)
    print(f'Nearest neighbor: {nearest.point}')

    k = 5
    k_neighbors = tree.k_nearest(target_point, k)
    print(f'\nK Nearest Neighbors (k={k}):')
    for dist2, node in k_neighbors:
        print(f'Point: {node.point}, Dist2: {dist2}')

    r = 10.0
    r_neighbors = tree.radius_search(target_point, r)
    print(f'\nNeighbors within radius (r={r})')
    for dist2, node in r_neighbors:
        print(f'Point: {node.point}, Dist2: {dist2}')
