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


from abc import ABC, abstractmethod
from typing import List, Tuple

class ConvexBody(ABC):
    '''This is an abstract class for convex bodies. '''

    def __init__(self, dim: int):
        """
        Initialize the ConvexBody with the given dimension.

        Parameters
        ----------
        dim : int
            The dimension of the convex body.
        """
        self.dim = dim

    @abstractmethod
    def plot(self, filename: str) -> None:
        """
        Plot the convex body and save it to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the plot.
        """
        pass

    @abstractmethod
    def is_point_on_surface(self, point: List[float], tolerance: float = 1e-6) -> bool:
        """
        Check whether the given point lies on the surface of the convex body.

        Parameters
        ----------
        point : list[float]
            A list of Cartesian coordinates of the point to be checked.
        tolerance : float, optional
            The tolerance to determine if the point lies on the surface of the convex body.
            The default value is 1e-6.

        Returns
        -------
        bool
            True if the point lies on the surface of the convex body, False otherwise.
        """
        pass

    @abstractmethod
    def volume(self) -> float:
        """
        Calculate the volume of the convex body.

        Returns
        -------
        float
            The volume of the convex body.
        """
        pass

    @abstractmethod
    def surface_area(self) -> float:
        """
        Calculate the surface area of the convex body.

        Returns
        -------
        float
            The surface area of the convex body.
        """
        pass

    @abstractmethod
    def centroid(self) -> List[float]:
        """
        Calculate the centroid of the convex body.

        Returns
        -------
        list[float]
            A list of Cartesian coordinates of the centroid.
        """
        pass

    @abstractmethod
    def bounding_box(self) -> Tuple[List[float], List[float]]:
        """
        Calculate the axis-aligned bounding box of the convex body.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing two lists of Cartesian coordinates: the minimum and maximum points of the bounding box.
        """
        pass
