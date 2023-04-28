from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from scipy.special import gamma
from scipy.stats import special_ortho_group
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

from base import ConvexBody


class LpBall(ConvexBody):
    """
    A class representing an LpBall in n-dimensional space.

    Attributes:
        dim (int): The dimension of the space.
        p (float): The parameter p of the Lp-norm.
        radius (float): The radius of the LpBall.
    """

    def __init__(self, n, p, radius=1):
        super().__init__(n)  # Call the parent class constructor
        self.n = n
        self.p = p
        self.radius = radius

    def is_point_on_surface(self, point: List[float], tolerance: float = 1e-8) -> bool:
        """
        Check if a point is on the surface of the LpBall.

        Parameters:
            point (list[float]): A point in the n-dimensional space.
            tolerance (float, optional): A tolerance value for determining if the point is on the surface.

        Returns:
            bool: True if the point is on the surface, False otherwise.
        """
        return (
            np.abs(np.sum(np.abs(point) ** self.p) - self.radius**self.p) <= tolerance
        )

    def distance_from_surface(self, point: List[float]) -> float:
        """
        Compute the distance from the point to the surface of the LpBall.

        The distance is positive if the point is inside the LpBall and negative if it is outside.

        Parameters:
            point (list[float]): A point in the n-dimensional space.

        Returns:
            float: The distance from the point to the surface of the LpBall.
        """
        lp_norm = np.power(np.sum(np.power(np.abs(point), self.p)), 1 / self.p)
        return self.radius - lp_norm

    def sample_points_on_surface(self):
        """
        Draw a random point uniformly on the surface of the LpBall.

        Returns:
            numpy.ndarray: A random point on the surface of the LpBall.
        """
        # Generate a random point on the unit LpBall (i.e., radius=1).
        unit_lp_ball_point = np.random.randn(self.n)
        unit_lp_ball_point /= np.power(np.sum(np.power(np.abs(unit_lp_ball_point), self.p)), 1 / self.p)
        
        # Scale the point to the desired LpBall radius.
        return self.radius * unit_lp_ball_point
    
    def plot(self, filename: str, num_points: int = 1000):
        """
        Plot the LpBall in 2D space.

        Parameters:
            filename (str): The name of the file to save the plot.
            num_points (int, optional): The number of points used to plot the LpBall. Default is 1000.
        """
        if self.dim != 2:
            raise NotImplementedError(
                "Plotting is currently only supported for 2D LpBalls."
            )

        # Generate points on the LpBall
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        points = np.column_stack((x, y))
        points_on_surface = np.power(np.abs(points), self.p - 1) * np.sign(points)
        points_on_surface /= np.power(
            np.sum(np.abs(points_on_surface) ** self.p, axis=1), 1 / self.p
        )[:, np.newaxis]

        # Plot the LpBall
        plt.figure()
        plt.plot(points_on_surface[:, 0], points_on_surface[:, 1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"LpBall with p={self.p} and radius={self.radius}")

        # Save the plot to a file
        plt.savefig(filename)
        plt.clf()

    def volume(self) -> float:
        if self.p == 2:
            # Use the formula for the volume of an n-dimensional Euclidean ball
            return (np.pi ** (self.dim / 2) * self.radius**self.dim) / np.prod(
                [np.math.gamma((i + 1) / 2) for i in range(1, self.dim * 2, 2)]
            )
        else:
            # Use Monte Carlo approximation for the volume
            num_samples = 100000
            points = np.random.uniform(
                -self.radius, self.radius, size=(num_samples, self.dim)
            )
            points_in_ball = (
                np.power(np.sum(np.abs(points) ** self.p, axis=1), 1 / self.p)
                <= self.radius
            )
            fraction_in_ball = np.mean(points_in_ball)
            bounding_box_volume = (2 * self.radius) ** self.dim
            return fraction_in_ball * bounding_box_volume

    def surface_area(self) -> float:
        # Use Monte Carlo approximation for the surface area
        num_samples = 100000
        points = np.random.uniform(
            -self.radius, self.radius, size=(num_samples, self.dim)
        )
        distances = np.power(np.sum(np.abs(points) ** self.p, axis=1), 1 / self.p)
        points_on_surface = np.isclose(distances, self.radius, rtol=1e-2)
        fraction_on_surface = np.mean(points_on_surface)
        bounding_box_volume = (2 * self.radius) ** self.dim
        return fraction_on_surface * bounding_box_volume

    def centroid(self) -> List[float]:
        # The centroid of an LpBall is at the origin
        return [0] * self.dim

    def bounding_box(self) -> Tuple[List[float], List[float]]:
        # The bounding box of an LpBall is given by its radius in each dimension
        return ([-self.radius] * self.dim, [self.radius] * self.dim)

    def __repr__(self) -> str:
        """
        A string representation of the LpBall.

        Returns:
            str: The string representation of the LpBall.
        """
        return f"LpBall(dim={self.dim}, p={self.p}, radius={self.radius})"
