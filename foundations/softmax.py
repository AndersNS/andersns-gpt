import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        c = np.empty(z.size, dtype=np.float64) 

        for index, x in np.ndenumerate(z):
            exp = np.exp(-x)
            c[index] = np.round(1 / (1 + exp),5)
        return c

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        c = np.empty(z.size, dtype=np.float64) 

        for index, x in np.ndenumerate(z):
            c[index] = np.max([0, x])
        return c
