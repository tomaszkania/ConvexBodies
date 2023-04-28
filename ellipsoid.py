from typing import Callable, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
from scipy.special import gamma


from base import ConvexBody


class Ellipsoid(ConvexBody):
    """This is a class for n-dimensional ellipsoids, inheriting from the ConvexBody abstract class."""

    def __init__(self, radii: List[float]):
        dim = len(radii)
        super().__init__(dim)
        self.radii = np.array(radii)
        self.coordinates = [self._coordinate(i) for i in range(self.dim)]

    def __repr__(self):
        """
        A string representation of the ellipsoid.

        :return: str, the string representation of the ellipsoid
        """
        return f"Ellipsoid(radii={self.radii})"

    def _coordinate(self, i):
        # Define the coordinate functions for the ellipsoid
        def coordinate(u, v):
            if i == 0:
                return self.radii[0] * np.cos(u) * np.sin(v)
            elif i == 1:
                return self.radii[1] * np.sin(u) * np.sin(v)
            elif i == self.dim - 1:
                return self.radii[-1] * np.cos(v)
            else:
                return self.radii[i] * np.sin(u) * np.cos(v)

        return coordinate

    def is_point_on_surface(self, point: List[float], tolerance: float = 1e-6) -> bool:
        """
        Check whether the given point lies on the surface of the ellipsoid.

        Parameters
        ----------
        point : list[float]
            A list of Cartesian coordinates of the point to be checked.
        tolerance : float, optional
            The tolerance to determine if the point lies on the surface of the ellipsoid.
            The default value is 1e-6.

        Returns
        -------
        bool
            True if the point lies on the surface of the ellipsoid, False otherwise.
        """
        # Compute the value of the ellipsoid equation for the given point:
        # (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
        ellipsoid_value = (
            sum((point[i] / self.radii[i]) ** 2 for i in range(len(self.radii))) - 1
        )

        # Check if the ellipsoid value is within the tolerance.
        return abs(ellipsoid_value) <= tolerance

    def volume(self) -> float:
        """
        Calculate the volume of the ellipsoid.

        Returns
        -------
        float
            The volume of the ellipsoid.
        """
        return (
            np.prod(self.radii)
            * np.pi ** (self.dim / 2)
            / np.prod([gamma((i + 1) / 2) for i in range(1, self.dim * 2, 2)])
        )

    def surface_area(self) -> float:
        """
        Calculate the surface area of the ellipsoid.

        Returns
        -------
        float
            The surface area of the ellipsoid.
        """
        if self.dim == 2:
            a, b = self.radii
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        elif self.dim == 3:
            a, b, c = self.radii
            p = 1.6075  # This value provides a good approximation; adjust as needed for more accuracy
            return (
                4
                * np.pi
                * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
            )
        else:
            raise NotImplementedError(
                "Surface area calculation is only implemented for 2D and 3D ellipsoids."
            )

    def centroid(self) -> List[float]:
        """
        Calculate the centroid of the ellipsoid.

        Returns
        -------
        list[float]
            A list of Cartesian coordinates of the centroid.
        """
        return [0.0] * self.dim

    def bounding_box(self) -> Tuple[List[float], List[float]]:
        """
        Calculate the axis-aligned bounding box of the ellipsoid.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing two lists of Cartesian coordinates: the minimum and maximum points of the bounding box.
        """
        min_coords = [-radius for radius in self.radii]
        max_coords = [radius for radius in self.radii]
        return min_coords, max_coords

    def hypersurface_measure(self, k):
        """
        Calculate the k-dimensional hypersurface measure of the ellipsoid.

        :param k: int, dimension of the hypersurface (1 <= k <= n)
        :return: float, the k-dimensional hypersurface measure
        """
        if k == 0:
            return 1
        elif k == self.dim:
            return (
                np.prod(self.radii)
                * np.pi ** (self.dim / 2)
                / np.prod([gamma((i + 1) / 2) for i in range(1, self.dim * 2, 2)])
            )
        elif self.dim == 2 and k == 1:
            # For a 2D ellipse, k must be 1, and we're calculating the perimeter
            # Use the Ramanujan approximation
            a, b = self.radii
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        elif self.dim == 1 and k == 1:
            # For a 1D ellipse, k must be 1, and we're calculating the length
            return 2 * self.radii[0]
        else:
            raise NotImplementedError(
                "Hypersurface measures for k != n are only implemented for 1D and 2D ellipsoids."
            )

    def surface_area_element(self, u: float, v: float) -> float:
        """
        Calculate the surface area element for the given parameter values u and v.

        Parameters:
        -----------
            u (float): Scalar parameter value u.
            v (float): Scalar parameter value v.

        Returns:
        --------
            float: The surface area element.
        """
        metric_tensor = np.zeros((self.dim - 1, self.dim - 1))
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                partial_i = self.partial_derivative(self.coordinates[i], u, v)
                partial_j = self.partial_derivative(self.coordinates[j], u, v)
                metric_tensor[i, j] = np.dot(partial_i, partial_j)

        return np.sqrt(np.linalg.det(metric_tensor))

    def partial_derivative(
        self,
        coordinate: Callable[[float, float], float],
        u: float,
        v: float,
        eps: float = 1e-5,
    ) -> Tuple[float, float]:
        """
        Calculate the partial derivatives of the given coordinate function with respect to u and v.

        Parameters:
        -----------
            coordinate (function): The coordinate function.
            u (float): Scalar parameter value u.
            v (float): Scalar parameter value v.
            eps (float): The step size for numerical differentiation (optional, default: 1e-5).

        Returns:
        --------
            tuple: The partial derivatives with respect to u and v.
        """
        du = (coordinate(u + eps, v) - coordinate(u - eps, v)) / (2 * eps)
        dv = (coordinate(u, v + eps) - coordinate(u, v - eps)) / (2 * eps)
        return du, dv

    def sample_points_on_surface(self, k: int) -> np.ndarray:
        """
        Sample k points uniformly on the surface of the ellipsoid.

        Parameters:
        -----------
            k (int): The number of points to sample.

        Returns:
        --------
            np.ndarray: A (k, n) array of k points on the surface of the ellipsoid.
        """
        # Sample points on the surface of an n-dimensional unit sphere
        points = self._sample_points_on_unit_sphere(k)

        # Scale the points by the radii of the ellipsoid
        scaled_points = points * self.radii

        return scaled_points

    def _sample_points_on_unit_sphere(self, k):
        """
        Sample k points uniformly on the surface of an n-dimensional unit sphere.

        Parameters:
        -----------
         
            k (int), the number of points to sample.
        
        Returns:
        --------
            np.array, a (k, n) array of k points on the surface of the unit sphere.
        """
        points = np.random.normal(size=(k, self.dim))
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        return points

    def plot(self, ax: Optional[plt.Axes] = None) -> None:
        if ax is None:
            fig = plt.figure()
            if self.dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)

        if self.dim == 2:
            self._plot_2d(ax)
        elif self.dim == 3:
            self._plot_3d(ax)
        else:
            raise NotImplementedError(
                "Plotting is only supported for 2D and 3D ellipsoids."
            )



    def _plot_2d(self) -> None:
        """
        Plots a 2D ellipsoid and save the plot to a file.

        Parameters:
        -----------
            filename (str): The name of the file to save the plot.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.radii[0] * np.cos(theta)
        y = self.radii[1] * np.sin(theta)

        plt.plot(x, y)
        plt.gca().set_aspect("equal", adjustable="box")
        
        plt.clf()


    def _plot_3d(self, ax: plt.Axes) -> None:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v))
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v))
        z = self.radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color="b", alpha=0.6)


