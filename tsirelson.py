from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from base import ConvexBody

"""
This module implements the computation of the Tsirelson norm for a given sequence.
The Tsirelson norm is a Banach space norm based on the work of Figiel and Johnson,
as described in the following paper:

    Figiel, T.; Johnson, W. B. (1974), "A uniformly convex Banach space which contains no $\ell_p$",
    Compositio Mathematica, 29: 179--190.

The implementation is inspired by the work of Michael Holt available at
https://github.com/holtm16/HonorsThesis.

The Tsirelson norm is computed using a recursive procedure that involves the Schreier
sets and a sequence of norms. The zeroth level norm is the maximum absolute value of
the elements in the sequence. The first level norm is the maximum of the zeroth level
norm and half the Schreier norm. For higher level norms, the norm is recursively
computed as the maximum of the previous level norm and half the Schreier norm
computed with respect to the current level.
"""


class Tsirelson(ConvexBody):
    def __init__(self, n, radius=1, theta=0.5) -> None:
        """
        Initialises the Tsirelson space with dimension n and given radius.

        Parameters:
        -----------
        n (int): the dimension of the space.
        radius (float): the radius of the ball to consider (defaults to 1).
        theta (float): the parameter theta of the Tsirelson space (defaults to 0.5).
        """
        self.n = n
        self.radius = radius
        self.theta = theta

    def __repr__(self):
        """
        A string representation of the Tsirelson space.

        Returns:
        str, the string representation of the Tsirelson space ball.
        """
        return f"Tsirelson(n={self.n}, radius={self.radius})"

    def tsirelson_norm(self, seq, level):
        def find_nth_schreier_norm(seq, n):
            """
            Computes the n-th Schreier norm of the input sequence.

            Parameters:
            -----------
                seq (list): array-like input sequence of numbers.
                n (int): Level of the desired Schreier norm.

            Returns:
            --------
                float, the n-th Schreier norm of the input sequence.
            """
            if n == 1:
                return find_schreier_norm(seq)

            N = len(seq)
            schreier_1_level_n = find_schreier_sets(N)
            max_value = 0

            for element_of_schreier_1_level_n in schreier_1_level_n:
                total_sum = 0

                array_of_sequences = make_interval_sequences(
                    seq, element_of_schreier_1_level_n
                )

                for new_seq in array_of_sequences:
                    total_sum += find_nth_level_norm(new_seq, n - 1)

                if total_sum > max_value:
                    # Update the max value
                    max_value = total_sum

            return max_value

        def make_interval_sequences(
            seq: List[float], element: Tuple[int]
        ) -> List[List[float]]:
            """
            Create interval sequences for the given input sequence and element of the Schreier set.

            Parameters:
            -----------
                seq (list): array-like input sequence of numbers.
                element (int): an element of the Schreier set.

            Returns:
                A list containing the created interval sequences.
            """
            list_of_new_seq = []
            N = len(seq) + 1
            last = len(element) - 1

            # Iterate through the element tuple, except for the last element
            for i in range(last):
                temp_1 = element[i]
                temp_2 = element[i + 1]
                # Create new sequences with leading zeroes and a subsequence from the input sequence
                new_seq = [0] * (temp_1 - 1) + seq[temp_1 - 1 : temp_2 - 1]
                list_of_new_seq.append(new_seq)

            # Create the last sequence where all values in seq are zeroed
            # out other than from the last index in element on
            last_new_seq = [0] * (element[last] - 1) + list(seq[element[last] - 1 :])

            list_of_new_seq.append(last_new_seq)

            return list_of_new_seq

        def find_nth_schreier_norm(seq: List[float], n: int) -> float:
            """
            Computes the n-th Schreier norm of the input sequence.

            Parameters:
            -----------
                seq (list): array-like input sequence of numbers.
                n (int): level of the desired Schreier norm.

            Returns:
            --------
                float, the n-th Schreier norm of the input sequence.
            """
            if n == 1:
                return find_schreier_norm(seq)

            N = len(seq)
            schreier_sets = find_schreier_sets(N)
            max_value = 0

            for schreier_set in schreier_sets:
                total_sum = 0
                interval_sequences = make_interval_sequences(seq, schreier_set)

                for new_seq in interval_sequences:
                    total_sum += find_nth_level_norm(new_seq, n - 1)

                if total_sum > max_value:
                    # Update the max value
                    max_value = total_sum

            return max_value

        def find_zero_norm(seq: List[float]) -> float:
            """
            Computes the zeroth norm of the input sequence.

            Parameters:
            -----------
                seq (list): array-like input sequence of numbers.

            Returns:
            --------
                float, the zeroth norm of the input sequence.
            """
            max_value = 0
            for element in seq:
                abs_value = abs(element)
                if abs_value > max_value:
                    # Update the max value
                    max_value = abs_value
            return max_value

        def find_schreier_norm(seq: List[float]) -> float:
            """
            Computes the Schreier norm of the input sequence.

            Parameters:
            -----------
                seq (list): array-like input sequence of numbers.

            Returns:
            --------
                float, the Schreier norm of the input sequence.
            """
            max_value = 0
            # Compute the Schreier sets for the given sequence length
            schreier_sets = find_schreier_sets(len(seq))

            # Iterate through the elements of the Schreier sets
            for schreier_set in schreier_sets:
                total_sum = 0
                # Compute the sum of the absolute values of the sequence elements corresponding to the indices in the Schreier set
                for index in schreier_set:
                    total_sum += abs(seq[index - 1])

                # Update the maximum value if the current sum is greater
                if total_sum > max_value:
                    max_value = total_sum

            return max_value

        def find_schreier_sets(N: int) -> List[Tuple[int]]:
            """
            Computes the significant Schreier sets for a sequence of length N.

            Parameters:
            -----------
                N (int): the length of the sequence.

            Returns:
            --------
                list: a list of tuples, each containing the significant Schreier sets for a sequence of size N.
            """
            if N == 0:
                return []
            elif N == 1:
                return [tuple([1])]
            elif N == 2:
                return [tuple([1]), tuple([2])]

            schreier_sets = [tuple([1])]
            temp_list = [i for i in range(1, N + 1)]

            # Adding the necessary elements of the Schreier Set to schreier_sets
            for i in range(2, int(N / 2 + 1)):
                # Using itertools.combinations() to obtain i length subsequences of temp_list[i-1:]
                list_of_tuples = list(combinations(temp_list[i - 1 :], i))
                for a_tuple in list_of_tuples:
                    schreier_sets.append(a_tuple)

            # For natural numbers greater than or equal to the floor M of N/2 + 1,
            # only the subset [M, M+1, M+2, ..., N] is considered.
            # This subset is in the Schreier family, and proper subsets of this subset do not need to be considered
            # since we only consider absolute values of elements in a given sequence x.
            M = int(N / 2 + 1)
            temp_subset = (M,)
            for j in range(M + 1, N + 1):
                temp_subset += (j,)

            if temp_subset not in schreier_sets:
                schreier_sets.append(temp_subset)

            return schreier_sets

        def find_first_level_norm(seq: List[float]) -> float:
            """
            Computes the first level norm of the input sequence.

            Parameters:
            -----------
                seq (list): Input sequence of numbers.

            Returns:
            --------
                float: the first level norm of the input sequence.
            """
            zero_norm_value = find_zero_norm(seq)
            half_schreier_norm_value = self.theta * find_schreier_norm(seq)
            if zero_norm_value >= half_schreier_norm_value:
                return zero_norm_value
            else:  # half_schreier_norm_value > zero_norm_value
                return half_schreier_norm_value

        def find_nth_level_norm(seq: List[float], n: int) -> float:
            """
            Computes the n-th level norm of the input sequence.

            Parameters:
            -----------
                seq (list): an array-like sequence of numbers.
                n (int): Level of the desired norm.

            Returns:
            --------
                float: the n-th level norm of the input sequence.
            """
            if seq == []:
                return 0
            elif n == 0:
                return find_zero_norm(seq)
            elif n == 1:
                return find_first_level_norm(seq)
            else:  # For n >= 2
                nth_norm_value = find_nth_level_norm(seq, n - 1)
                half_schreier_norm_value = self.theta * find_nth_schreier_norm(seq, n)

                if nth_norm_value >= half_schreier_norm_value:
                    return nth_norm_value
                else:  # half_schreier_norm_value > nth_norm_value
                    return half_schreier_norm_value

        return find_nth_level_norm(seq, level)

    def is_point_on_surface(self, point, radius=None):
        """
        Check if a given point is on the surface of the Tsirelson space.

        :param point: array-like, a list or a numpy array of n elements
        :param radius: float, the radius of the LpBall (default: None)
        :return: bool, True if the point is on the surface, False otherwise
        """
        if radius is None:
            radius = self.radius

        norm_value = self.tsirelson_norm(point)
        return np.isclose(norm_value, radius, atol=1e-6)

    def sample_points_on_surface(self, num_points, level):
        points = []
        for _ in range(num_points):
            point = np.random.randn(self.n)
            point /= self.tsirelson_norm(point, level)
            point *= self.radius
            points.append(point)
        return np.array(points)

    def plot(self, filename=None):
        """
        Plot the Tsirelson space.

        :param filename: str or None, the filename to save the plot (default: None)
        """
        if self.n == 2:
            fig, ax = plt.subplots()
            points = self.sample_points_on_surface(10000, 2)
            ax.scatter(points[:, 0], points[:, 1], s=1)
        elif self.n == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            points = self.sample_points_on_surface(80000, 3)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        else:
            raise NotImplementedError("Plotting is only supported for 2D and 3D cases.")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def bounding_box(self):
        """
        Calculate the bounding box of the Tsirelson space.

        :return: tuple, the minimum and maximum points of the bounding box
        """
        min_point = -self.radius * np.ones(self.n)
        max_point = self.radius * np.ones(self.n)
        return min_point, max_point

    def centroid(self):
        """
        Calculate the centroid of the Tsirelson space.

        :return: numpy.ndarray, the centroid of the Tsirelson space
        """
        # The centroid of the Tsirelson space is the origin.
        return np.zeros(self.n)

    def is_point_on_surface(self, point, tolerance=1e-8):
        """
        Check if a given point is on the surface of the Tsirelson space.

        :param point: numpy.ndarray, the point to check
        :param tolerance: float, the tolerance for comparing the norm (default: 1e-8)
        :return: bool, True if the point is on the surface, False otherwise
        """
        norm = self.tsirelson_norm(point)
        return np.abs(norm - self.radius) < tolerance

    def surface_area(self):
        """
        Calculate the surface area of the Tsirelson space.

        :return: float, the surface area of the Tsirelson space
        """

        raise NotImplementedError("Surface area is to be implemented.")

    def volume(self):
        """
        Calculate the volume of the Tsirelson space.

        :return: float, the volume of the Tsirelson space
        """

        raise NotImplementedError("Volume is to be implemented.")
