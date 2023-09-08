import numpy as np
from scipy.spatial.distance import cdist
from colour.algebra import table_interpolation_tetrahedral
from colour import LUT3D, read_LUT
from .Node import Node

from typing import Optional, Any
from typing import Optional, Any, Tuple
from numpy.typing import NDArray
# xalglib only available in Windows & Linux(?)
try:
    from xalglib import xalglib
except ImportError:
    import warnings
    warnings.warn("RBF library cannot be loaded.", ImportWarning)

class RBF(Node):
    def __init__(self, radius: float=1):
    def __init__(self, size: int=33, radius: float=5.0, layers: int=10, smoothing: float=0.001):
        self.size = size
        self.LUT = None

        self.radius = radius
        self.weights = None
        self.coordinates = None
        self.layers = layers
        self.smoothing = smoothing

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        data = np.hstack((source, target))

        model = xalglib.rbfcreate(3, 3)
        xalglib.rbfsetpoints(model, data.tolist())
        xalglib.rbfsetalgohierarchical(model, self.radius, self.layers, self.smoothing)
        xalglib.rbfbuildmodel(model)

        grid = np.linspace(0, 1, self.size).tolist()
        table = xalglib.rbfgridcalc3v(model, grid, self.size, grid, self.size, grid, self.size)

        # xalglib outputs coordinates in (z, y, x). Swapping axis 0 and 2
        # gives (x, y, z) which is needed for the LUT table.
        LUT_table = np.reshape(table, (self.size, self.size, self.size, 3)).swapaxes(0, 2)

        self.LUT = LUT3D(table=LUT_table)
        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)

    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        self.coordinates = source
        self.weights = self._solve_weights(source, target)
class LUT(Node):
    def __init__(self, path):
        self.LUT = read_LUT(path)

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.weights is None or self.coordinates is None:
        if self.LUT is None:
            return RGB

        shape = RGB.shape
        RGB = np.reshape(RGB, (-1, 3))

        points = self.coordinates.shape[0]

        H = np.zeros((RGB.shape[0], points + 3 + 1))
        H[:, :points] = self.basis(cdist(RGB, self.coordinates), self.radius)
        H[:, points] = 1.0
        H[:, -3:] = RGB
        return np.reshape(np.asarray(np.dot(H, self.weights)), shape)

    def _solve_weights(self, X, Y):
        npts, dim = X.shape
        H = np.zeros((npts + 3 + 1, npts + 3 + 1))
        H[:npts, :npts] = self.basis(cdist(X, X), self.radius)
        H[npts, :npts] = 1.0
        H[:npts, npts] = 1.0
        H[:npts, -3:] = X
        H[-3:, :npts] = X.T

        rhs = np.zeros((npts + 3 + 1, dim))
        rhs[:npts, :] = Y
        return np.linalg.solve(H, rhs)

    @staticmethod
    def basis(X, r):
        arg = X / r
        v = 1 - arg / 9
        return np.where(v > 0, np.exp(1 - arg - 1/v), 0)
        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)
