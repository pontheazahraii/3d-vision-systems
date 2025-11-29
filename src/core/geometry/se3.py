'''
An SE(3) transformation represents a rigid body transformation in 3D space combining both rotation and translation. It is typically represented as a 4x4 homogenous transformation matrix.
'''

import numpy as np


class SE3Transform:
    def __init__(self, R=None, t=None):
        '''
        Initialize an SE(3) transformation.

        Parameters:
            R (np.ndarray, optional):
                3x3 rotation matrix (SO(3)). If None, an identity matrix is used.
            t (np.ndarray, optional):
                3x1 translation vector. If None, a zero vector is used.

        Raises:
            ValueError:
                If `R` is not 3x3.
            ValueError:
                If `t` is not 3x1.
        '''
        if R is None:
            self.R = np.eye(3)
        else:
            if R.shape != (3, 3):
                raise ValueError("Rotation matrix R must be 3x3.")
            self.R = R

        if t is None:
            self.t = np.zeros((3, 1))
        else:
            if t.shape != (3, 1):
                raise ValueError("Translation vector t must be 3x1.")
            self.t = t

        self.H = self.__to_homogenous_matrix()

    def __to_homogenous_matrix(self):
        '''Converts the rotation and translation to a 4x4 homogenous matrix.'''
        H = np.eye(4)
        H[:3, :3] = self.R
        H[:3, 3] = self.t.flatten()
        return H

    @classmethod
    def from_homogenous_matrix(cls, H):
        '''
        Creates an SE3Transformation from a 4x4 homogenous matrix

        Parameters:
            H (np.ndarray):
                4x4 homogenous matrix.

        Raises:
            ValueError:
                If `H` is not 4x4.
        '''
        if H.shape != (4, 4):
            raise ValueError('Homogenous matrix H must be 4x4.')
        R = H[:3, :3]
        t = H[:3, 3].reshape((3, 1))
        return cls(R, t)

    def inverse(self):
        '''
        Computes the inverse of the SE(3) transformation.

        Returns:
            SE3Transform:
                A new SE3Transform instance representing the inverse of the current transformation.
        '''
        R_inv = self.R.T
        t_inv = -np.dot(R_inv, self.t)
        return SE3Transform(R_inv, t_inv)

    def transform_point(self, point):
        '''
        Transform a 3D point (represented by a 3x1 vector).

        Parameters:
            point (np.ndarray):
                3x1 NumPy array representing the point.

        Returns:
            np.ndarray:
                The transformed 3D point

        Raises:
            ValueError:
                If `point` is not 3x1.
        '''
        if point.shape != (3, 1):
            raise ValueError("Point must be a 3x1 vector.")

        return np.dot(self.R, point) + self.t

    def __mul__(self, other):
        '''
        Composes two SE(3) transformations or transforms a point.

        Parameters:
            other (SE3Transform, np.ndarray):
                Either another SE3Transform to compose with, or a 3x1 numpy array representing a 3D point to transform.

        Returns:
            SE3Transform or np.ndarray:
                - `SE3Transform` if composing two transformations.
                - `(3, 1) np.ndarray` if transforming a point.

        Raises:
            TypeError:
                If `other` is neither an SE3Transform nor a (3, 1) numpy array.
        '''
        if isinstance(other, SE3Transform):
            new_H = np.dot(self.H, other.H)
            return SE3Transform.from_homogenous_matrix(new_H)
        elif isinstance(other, np.ndarray) and other.shape == (3, 1):
            return self.transform_point(other)
        else:
            raise TypeError('Unsupported operand type of multiplication.')

    def __str__(self):
        return f'SE(3) Transform:\nRotation:\n{self.R}\nTranslation:\n{self.t}'


# Example usage
if __name__ == '__main__':
    # create a rotation matrix (e.g., 90 degree)
    theta = np.pi / 2
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # create a translation vector
    t_vec = np.array([[1.0], [2.0], [3.0]])

    # Create an SE(3) transform
    transform_x = SE3Transform(R=R_x, t=t_vec)
    transform_y = SE3Transform(R=R_y, t=t_vec)
    transform_z = SE3Transform(R=R_z, t=t_vec)

    print(f'Transform x:\n{transform_x}')
    print(f'\nTransform y:\n{transform_y}')
    print(f'\nTransform z:\n{transform_z}')

    # Create another SE(3) transform (e.g. pure translation)
    transform_pure = SE3Transform(t=np.array([[-1.0], [0.75], [0.0]]))
    print(f'\nPure Transform:\n{transform_pure}')

    # Compose transformations
    composed_transform = transform_y * transform_pure
    print(f'\nComposed Transform (Transform y * Pure Transform):\n{composed_transform}')

    # Transform a point
    point_A = np.array([[1.0], [0.0], [0.0]])
    transformed_point = transform_y * point_A
    print(f'\nOriginal Point A:\n{point_A}')
    print(f'\nTransformed Point A by Transform 1:\n{transformed_point}')

    # Inverse transformation
    inv_transform_y = transform_y.inverse()
    print(f'\nInverse of Transform y:\n{inv_transform_y}')

    # Verify inverse
    recovered_point_A = inv_transform_y * transformed_point
    print(f'\nOriginal Point A recovered by inverse transformation:\n{recovered_point_A}')
