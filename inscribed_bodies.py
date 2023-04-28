
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from scipy.special import gamma
from scipy.stats import special_ortho_group
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from functools import partial
from itertools import product
from itertools import combinations

from ellipsoid import Ellipsoid
from parallelepiped import Parallelepiped
from lpball import LpBall

class ParallelepipedInscribedInEllipsoid(Ellipsoid):
    def __init__(self, radii):
        """
        Initialize the inscribed parallelepiped with the given ellipsoid radii and center.

        :param radii: array-like, a list or a numpy array of n radii for the ellipsoid
        :param center: array-like, a numpy array representing the center of the ellipsoid, optional
        """
        super().__init__(radii)
        self.dim = len(radii)
        self.ellipsoid = Ellipsoid(radii)

        # Sample a random point on the ellipsoid surface
        origin = self.sample_points_on_surface(1)[0]

        # Generate the remaining points on the surface
        points = self._generate_inscribed_points(origin)

        # Compute the parallelepiped vectors and create a Parallelepiped instance
        vectors = [points[i] - origin for i in range(len(radii))]
        self.parallelepiped = Parallelepiped(vectors, origin)

    def __repr__(self):
        """
        A string representation of the inscribed parallelepiped.

        :return: str, the string representation of the inscribed parallelepiped
        """
        return f"InscribedParallelepiped(ellipsoid={self.ellipsoid}, parallelepiped={self.parallelepiped})"
        
    

    def _generate_inscribed_points(self, origin):
        """
        Generate the remaining points on the surface of the ellipsoid for the inscribed parallelepiped.

        :param origin: array-like, a numpy array representing the origin point
        :return: list, a list of numpy arrays representing the remaining inscribed points
        """
        points = []

        for _ in range(self.dim):
            # Sample a point on the ellipsoid surface and compute its reflection
            point = self.sample_points_on_surface(1)[0]
            reflection = 2 * origin - point

            # Add the reflected point to the list of points
            points.append(reflection)

        return points

        
    

    def hypersurface_measure(self, k):
        """
        Calculate the k-dimensional hypersurface measure of the inscribed parallelepiped.

        :param k: int, the dimension of the hypersurface measure
        :return: float, the k-dimensional hypersurface measure
        """
        return self.parallelepiped.hypersurface_measure(k)

    def total_edge_circumference(self):
        """
        Compute the total circumference of the edges of the inscribed parallelepiped.

        :return: float, the total circumference of the edges
        """
        return self.parallelepiped.total_edge_circumference()

    
    
    @classmethod
    def maximize_edge_circumference(cls, radii, origin, ellipsoid=None):
        def objective(points):
            points = points.reshape(-1, len(radii))
            vectors = [points[i] - origin for i in range(len(radii))]
            parallelepiped = Parallelepiped(vectors, origin)
            return -parallelepiped.total_edge_circumference()

        def constraint(points):
            points = points.reshape(-1, len(radii))
            return [np.linalg.norm(point - origin) - radius for point, radius in zip(points, radii)]

        if ellipsoid is None:
            ellipsoid = Ellipsoid(radii)

        # Use the surface points of the ellipsoid as initial points for optimization
        surface_points = ellipsoid.sample_points_on_surface(len(radii))
        initial_points = np.array(surface_points)

        # Optimize the objective function with the Nelder-Mead method
        result = minimize(
            objective,
            initial_points.flatten(),
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint},
            options={"maxiter": 1000, "ftol": 1e-4},
        )

        # Create an inscribed parallelepiped with the optimized points
        optimized_points = result.x.reshape(-1, len(radii))
        vectors = [optimized_points[i] - origin for i in range(len(radii))]
        parallelepiped = Parallelepiped(vectors, origin)

        return parallelepiped

    
    @classmethod
    def maximize_surface_area(cls, radii, origin, ellipsoid=None):
        def objective(points):
            points = points.reshape(-1, len(radii))
            vectors = [points[i] - origin for i in range(len(radii))]
            parallelepiped = Parallelepiped(vectors, origin)
            return -parallelepiped.hypersurface_measure(2)

        def constraint(points):
            points = points.reshape(-1, len(radii))
            return [np.linalg.norm(point - origin) - radius for point, radius in zip(points, radii)]

        if ellipsoid is None:
            ellipsoid = Ellipsoid(radii)

        # Use the surface points of the ellipsoid as initial points for optimization
        surface_points = ellipsoid.sample_points_on_surface(len(radii))
        initial_points = np.array(surface_points)

        # Optimize the objective function with the Nelder-Mead method
        result = minimize(
            objective,
            initial_points.flatten(),
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint},
            options={"maxiter": 1000, "ftol": 1e-4},
        )

        # Create an inscribed parallelepiped with the optimized points
        optimized_points = result.x.reshape(-1, len(radii))
        vectors = [optimized_points[i] - origin for i in range(len(radii))]
        parallelepiped = Parallelepiped(vectors, origin)

        return parallelepiped
    
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

        # Set axis limits
        for i in range(self.dim):
            ax.set_xlim(-self.radii[0], self.radii[0])
            ax.set_ylim(-self.radii[1], self.radii[1])
            if self.dim == 3:
                ax.set_zlim(-self.radii[2], self.radii[2])

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()



class ParallelepipedInscribedInLpBall(LpBall):
    
    def __init__(self, lp_ball, origin):
        self.lp_ball = lp_ball
        self.origin = np.array(origin)

    @staticmethod
    def _constraint(vectors_flat, lp_ball, origin):
        vectors = vectors_flat.reshape(-1, lp_ball.n)
        vertices = np.vstack([origin, origin + vectors])
        constraint_values = np.array([lp_ball.distance_from_surface(vertex) for vertex in vertices])
        return constraint_values

    @classmethod
    def maximize_edge_circumference(cls, lp_ball, origin):
        
        def objective(vectors_flat, lp_ball, origin):
            vectors = vectors_flat.reshape(-1, lp_ball.n)
            parallelepiped = Parallelepiped(vectors, origin)
            return -parallelepiped.total_edge_circumference()

        initial_vectors = np.eye(lp_ball.n)
        initial_vectors_flat = initial_vectors.flatten()
        
        result = minimize(
            objective,
            initial_vectors_flat,
            args=(lp_ball, origin),
            method="SLSQP",
            constraints={"type": "eq", "fun": cls._constraint, "args": (lp_ball, origin)},
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        optimized_vectors = result.x.reshape(-1, lp_ball.n)
        return Parallelepiped(optimized_vectors, origin)
    
    def _compute_inscribed_parallelepiped(self):
        # Compute the parallelepiped vertices
        vertices = self.lp_ball.get_vertices()
        self.vertices = vertices

        # Compute the vectors of the parallelepiped
        vectors = [vertices[i] - vertices[0] for i in range(1, 2*self.n)]

        # Create a new Parallelepiped object
        self.parallelepiped = Parallelepiped(vectors, vertices[0])

    def _compute_inscribed_parallelepiped(self):
        # Compute the parallelepiped vertices
        vertices = self.lp_ball.get_vertices()
        self.vertices = vertices

        # Compute the vectors of the parallelepiped
        vectors = [vertices[i] - vertices[0] for i in range(1, 2*self.n)]

        # Create a new Parallelepiped object
        self.parallelepiped = Parallelepiped(vectors, vertices[0])

    def plot(self, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the LpBall
        self.lp_ball.plot(ax=ax)

        # Plot the inscribed parallelepiped
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

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()