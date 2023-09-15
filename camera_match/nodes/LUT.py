import numpy as np
from typing import Tuple, Any
from scipy.spatial.distance import cdist
from colour.algebra import table_interpolation_tetrahedral
from colour import LUT3D, read_LUT
from .Node import Node
from tqdm import tqdm

from typing import Optional, Any, Tuple
from numpy.typing import NDArray
# xalglib only available in Windows & Linux(?)
try:
    from xalglib import xalglib
except (ImportError, OSError):
    import warnings
    warnings.warn("RBF library cannot be loaded.", ImportWarning)

class RBF(Node):
    def __init__(self, size: int=33, radius: float=5.0, layers: int=10, smoothing: float=0.001):
        self.size = size
        self.LUT = None
        self.radius = radius
        self.layers = layers
        self.smoothing = smoothing

    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        data = np.hstack((source, target))

        model = xalglib.rbfcreate(3, 3)
        xalglib.rbfsetpoints(model, data.tolist())

        # Part 1: Setting up the RBF model
        with tqdm(total=50, desc="Setting up RBF Model") as progress_bar:
            progress_bar.update(10)  # Update progress by 10% for the setup part

            xalglib.rbfsetalgohierarchical(model, self.radius, self.layers, self.smoothing)
            progress_bar.update(40)  # Update progress by 40% for hierarchical setup

        xalglib.rbfbuildmodel(model)

        # Part 2: Building the LUT table
        with tqdm(total=50, desc="Building LUT Table") as progress_bar:
            grid = np.linspace(0, 1, self.size).tolist()
            table = xalglib.rbfgridcalc3v(model, grid, self.size, grid, self.size, grid, self.size)

            # xalglib outputs coordinates in (z, y, x). Swapping axis 0 and 2
            # gives (x, y, z) which is needed for the LUT table.
            LUT_table = np.reshape(table, (self.size, self.size, self.size, 3)).swapaxes(0, 2)

            self.LUT = LUT3D(table=LUT_table)
            progress_bar.update(50)  # Update progress by 50% for table building

        # Close the progress bar
        progress_bar.close()

        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)

class LUT(Node):
    def __init__(self, path):
        self.LUT = read_LUT(path)
    
    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)
