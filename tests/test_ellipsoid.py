import numpy as np
from ..ellipsoid import Ellipsoid


def test_ellipse_perimeter():
    ellipse = Ellipsoid([3, 2])
    a, b = 3, 2
    h = ((a - b)**2) / ((a + b)**2)
    expected_perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))  # Ramanujan approximation
    actual_perimeter = ellipse.hypersurface_measure(1)
    np.testing.assert_almost_equal(actual_perimeter, expected_perimeter, decimal=5)

def test_circle_area():
    circle = Ellipsoid([5, 5])
    expected_area = np.pi * 5**2
    actual_area = circle.hypersurface_measure(2)
    np.testing.assert_almost_equal(actual_area, expected_area, decimal=5)


def test_sphere_surface_area():
    sphere = Ellipsoid([3, 3, 3])
    expected_surface_area = 4 * np.pi * 3**2
    actual_surface_area = sphere.hypersurface_measure(2)
    np.testing.assert_almost_equal(actual_surface_area, expected_surface_area, decimal=5)

def test_ellipsoid_volume():
    ellipsoid = Ellipsoid([2, 3, 4])
    expected_volume = 4/3 * np.pi * 2 * 3 * 4
    actual_volume = ellipsoid.hypersurface_measure(3)
    np.testing.assert_almost_equal(actual_volume, expected_volume, decimal=5)

