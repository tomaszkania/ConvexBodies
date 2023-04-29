# Convex Bodies

This repository contains a Python library for working with convex bodies, such as ellipsoids and parallelepipeds, in two and three dimensions. The library provides functionality for creating, transforming, and visualising these convex bodies, as well as computing their geometric properties, such as volume, surface area, and various norms.

## Features

- Create and manipulate ellipsoids and parallelepipeds in 2D and 3D
- Compute geometric properties, such as volume and surface area
- Calculate norms, including Tsirelson norms and Schreier norms
- Visualise the convex bodies using 2D plots and 3D surface plots
- Extensible design, making it easy to add support for additional convex bodies
- Encoding balls of finite-dimensional Lp- and Tsirelson-spaces.

## Installation

To install the library, simply clone the repository and add the `ConvexBodies` directory to your Python path:

```git clone https://github.com/yourusername/ConvexBodies.git```


Then, add the following line to your Python script to import the library:

```import ConvexBodies```

## Usage
### Ellipsoid

Create an ellipsoid (centred at origin) by specifying its radii:

```from ConvexBodies import Ellipsoid```

```# Create a 3D ellipsoid with radii 1, 2, and 3```

```ellipsoid = Ellipsoid([1, 2, 3])```

Compute properties of the ellipsoid:

```volume = ellipsoid.volume()```
```surface_area = ellipsoid.surface_area()```

Plot the ellipsoid:

```ellipsoid.plot()```

### Parallelepiped

Create a parallelepiped by specifying its edge vectors:

```from ConvexBodies import Parallelepiped```

```# Create a 3D parallelepiped with edge vectors (1, 0, 0), (0, 2, 0), and (0, 0, 3)```

```parallelepiped = Parallelepiped([(1, 0, 0), (0, 2, 0), (0, 0, 3)])```

Compute properties of the parallelepiped:

```volume = parallelepiped.volume()```

Plot the parallelepiped:

```parallelepiped.plot()```

## Contributing

Contributions to the project are welcome. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.
License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

The implementation of Tsirelson norms is based on the paper "A uniformly convex Banach space which contains no ℓ" by Figiel, T. and Johnson, W. B. (1974), Compositio Mathematica, 29: 179–190.
The code is inspired by the implementation of Michael Holt's Honors Thesis, available at https://github.com/holtm16/HonorsThesis

