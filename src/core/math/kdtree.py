'''
A KD-Tree (K-Dimensional Tree) is a space-partitioning data structure used to organize points in a k-dimensional space, facilitating efficient search operations like nearest neighbor queries
'''


class KDNode:
    def __init__(self, point, k, left=None, right=None):
        '''
        Initialize the node for the KD-Tree.

        Parameters:
            point (Tuple[float, ...] or List[float]):
                An array representing the coordinates in k-dimensional space.
                Must be indexable (e.g., tuple, list) and have length `k`.
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
    def __init__(self, points):
        '''
        Initializes and builds the KD-Tree.

        Parameters:
            points(List[Tuple[float, ...]] or List[List[float]]):
                A list of k-dimensional points.

        Raise:
            ValueError:
                If point list is empty.
        '''
        if not points:
            raise ValueError('Points list cannot be empty.')

        self.k = len(points[0])
        self.root = self._build(points)

    def _build(self, points, depth=0):
        '''Recursively builds the KD-Tree.'''
        if not points:
            return None

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
            target (Tuple[float, ...] or List[float]):
                The query point.

        Returns:
            KDNode:
                Node containing the nearest point.
        '''
        return self._nearest(self.root, target, depth=0, best=None)

    def _nearest(self, node, target, depth, best):
        '''Recursive helper for nearest neighbor search.'''
        if node is None:
            return best

        axis = depth % self.k

        # Update best if current node is closer
        if best is None or self._distance_squared(target, node.point) < self._distance_squared(target, best.point):
            best = node

        # Pick branch based on splitting axis
        if target[axis] < node.point[axis]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        # Search primary branch
        best = self._nearest(next_branch, target, depth+1, best)

        # Check if we need to explore the other branch
        if abs(target[axis] - node.point[axis])**2 < self._distance_squared(target, best.point):
            best = self._nearest(other_branch, target, depth+1, best)

        return best


if __name__ == '__main__':
    points = [(3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7)]
    tree = KDTree(points)

    target_point = (20, 17)
    nearest = tree.nearest(target_point)

    print("Nearest neighbor:", nearest.point)
