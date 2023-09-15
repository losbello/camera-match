import numpy as np
import threading  # Import threading module
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

        # Initialize the progress bar
        total_steps = 100
        progress_bar = tqdm(total=total_steps, desc="Progress")

        # Part 1: Setting up the RBF model
        with tqdm(total=50, desc="Setting up RBF Model") as sub_progress_bar:
            sub_progress_bar.update(10)
            xalglib.rbfsetalgohierarchical(model, self.radius, self.layers, self.smoothing)
            sub_progress_bar.update(40)

        def blocking_call():  # Define the blocking function
            print("Starting the blocking function...")
            try:
            xalglib.rbfbuildmodel(model)
            except Exception as e:
                print(f"An error occurred: {e}")
            print("Blocking function completed.")


        # Create a thread to run the blocking function
        thread = threading.Thread(target=blocking_call)

        # Start the thread
        thread.start()
        print("Thread started.")


        # Part 2: Building the LUT table
        print("About to start second progress bar")
        with tqdm(total=50, desc="Building LUT Table") as sub_progress_bar:
            grid = np.linspace(0, 1, self.size).tolist()
            table = xalglib.rbfgridcalc3v(model, grid, self.size, grid, self.size, grid, self.size)

            num_sub_steps = 10
            sub_step_size = len(table) // num_sub_steps

            for sub_step in range(num_sub_steps):
                start_index = sub_step * sub_step_size
                end_index = (sub_step + 1) * sub_step_size

                sub_table = table[start_index:end_index]
                sub_progress_bar.update(5)

            sub_progress_bar.update(50 - (5 * num_sub_steps))

        # Wait for the thread to complete
        thread.join()
        print("Main thread has joined the child thread.")

        # Close the progress bars
        progress_bar.close()
        sub_progress_bar.close()

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
