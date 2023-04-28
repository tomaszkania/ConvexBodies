
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
from scipy.special import gamma
from scipy.stats import special_ortho_group
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from functools import partial
from itertools import product
from itertools import combinations

from base import ConvexBody

class Parallelepiped(ConvexBody):
    def __init__(self, vectors, origin):
        """
        Initialize the parallelepiped with its n vectors and origin.

        :param vectors: array-like, a list of n numpy arrays representing the n-dimensional vectors
        :param origin: array-like, a numpy array representing the origin point
        """
        self.vectors = np.array(vectors)
        self.origin = np.array(origin)
        self.dim = len(vectors)
        
    def get_vertices(self):
        vertices = [self.origin]
        for i in range(1, 2 ** self.dim):
            binary_repr = format(i, f"0{self.dim}b")
            vertex = self.origin.copy()
            for j, bit in enumerate(binary_repr):
                if bit == '1':
                    vertex += self.vectors[j]
            vertices.append(vertex)
        return np.array(vertices)


    def plot(self, filename=None):
        if self.dim == 2:
            fig, ax = plt.subplots()
            super().plot(ax)
            vertices = self.parallelepiped.get_vertices()
            polygon = plt.Polygon(vertices[:, :2], edgecolor='k', alpha=0.5, linewidth=1)
            ax.add_patch(polygon)
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            super().plot(ax)

            vertices = self.parallelepiped.get_vertices()
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[1], vertices[3], vertices[2]],
                [vertices[4], vertices[5], vertices[7], vertices[6]],
                [vertices[0], vertices[2], vertices[6], vertices[4]],
                [vertices[1], vertices[3], vertices[7], vertices[5]]
            ]

            face_collection = Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=0.5)
            face_colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
            face_collection.set_facecolor(face_colors)
            ax.add_collection3d(face_collection)
        else:
            raise ValueError("Plotting is supported only for 2 and 3 dimensions")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if self.dim == 3:
            ax.set_zlabel('Z')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


    def hypersurface_measure(self, k):
        if k != 2:
            raise ValueError("Only 2-dimensional surface area calculation is supported")

        vector_combinations = list(combinations(self.vectors, k))

        # Calculate the sum of the absolute values of the determinants of all combinations
        hypersurface_measure = sum(
            [np.linalg.norm(np.cross(*comb)) for comb in vector_combinations]
        )

        return hypersurface_measure


    def total_edge_circumference(self):
        """
        Compute the total circumference of the edges of the parallelepiped.

        :return: float, the total circumference of the edges
        """
        # Calculate the length of each vector and sum them up
        edge_lengths = [np.linalg.norm(vector) for vector in self.vectors]
        return 2 * self.dim * sum(edge_lengths)

    def __repr__(self):
        """
        A string representation of the parallelepiped.
        """
        vectors_formatted = np.array2string(self.get_vertices(), formatter={'float_kind': lambda x: f'{x:.2f}'})
        return f"Parallelepiped(vertices={vectors_formatted}, origin={self.origin})"
    
    def is_point_on_surface(self, point, tolerance=1e-8):
        """
        Check if a point is on the surface of the parallelepiped.

        :param point: array-like, a numpy array representing the point to check
        :param tolerance: float, the tolerance for floating point comparisons

        :return: bool, True if the point is on the surface of the parallelepiped, False otherwise
        """
        vertices = self.get_vertices()
        # Check if the point is within the tolerance of any of the edges
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                edge = vertices[j] - vertices[i]
                projection = np.dot(point - vertices[i], edge) / np.dot(edge, edge)
                # Check if the projection is within the tolerance and the point is within the bounds of the edge
                if 0 <= projection <= 1 and np.linalg.norm(point - (vertices[i] + projection * edge)) <= tolerance:
                    return True
        return False
    
    def is_vertex(self, point, tolerance=1e-8):
        """
        Check if a point is a vertex of the parallelepiped.

        :param point: array-like, a numpy array representing the point to check
        :param tolerance: float, the tolerance for floating point comparisons

        :return: bool, True if the point is a vertex of the parallelepiped, False otherwise
        """
        vertices = self.get_vertices()
        for vertex in vertices:
            # Check if the distance between the point and a vertex is within the tolerance
            if np.linalg.norm(point - vertex) <= tolerance:
                return True
        return False
    
    def volume(self) -> float:
        # Calculate the volume using the absolute value of the determinant
        # of the matrix formed by the edge vectors.
        return np.abs(np.linalg.det(self.vectors))

    def surface_area(self) -> float:
        # Use the hypersurface_measure method for k = 2 to calculate the surface area
        return self.hypersurface_measure(k=2)

    def centroid(self) -> List[float]:
        # Calculate the centroid as the average of the vertices
        vertices = self.get_vertices()
        return list(np.mean(vertices, axis=0))

    def bounding_box(self) -> Tuple[List[float], List[float]]:
        # Calculate the minimum and maximum coordinates for each dimension
        vertices = self.get_vertices()
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (list(min_coords), list(max_coords))


    @staticmethod
    def parallelepiped_circumference(vertices: np.ndarray) -> float:
        """
        Compute the total circumference of the edges of a parallelepiped given its vertices.

        :param vertices: array-like, a list of numpy arrays representing the vertices of the parallelepiped
        :return: float, the total circumference of the edges
        """
        n = vertices.shape[1]
        edge_vectors = [vertices[2 ** i] - vertices[0] for i in range(n)]

        # Calculate the length of each edge vector and sum them up
        edge_lengths = [np.linalg.norm(vector) for vector in edge_vectors]
        total_circumference = 2 * n * sum(edge_lengths)

        return total_circumference

    @staticmethod
    def is_parallelepiped(points, tolerance=1e-8):
        points = np.array(points)
        n = points.shape[1]

        if points.shape[0] != 2 ** n:
            return False

        # Compute edge vectors
        edge_vectors = [points[2 ** i] - points[0] for i in range(n)]

        for i in range(1, 2 ** n):
            binary_repr = format(i, f"0{n}b")
            target_point = points[0] + np.sum(
                [edge_vectors[j] * int(binary_repr[j]) for j in range(n)], axis=0
            )

            found = False
            for point in points:
                if np.linalg.norm(target_point - point) < tolerance:
                    found = True
                    break

            if not found:
                return False

        return True