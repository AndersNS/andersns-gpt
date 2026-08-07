import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value


        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))  

        # Loss: L = 0.5 * (y_hat - y_true)^2

        # Gradient:
        error = y_hat - y_true
        sigmoid_deriv = y_hat * (1.0 - y_hat)
        
        # delta is the overall derivative of the loss function:
        dl_dz = error * sigmoid_deriv
        
        # dl_dw is the derivative of the loss function with respect to each weight
        dl_dw = x * dl_dz
        # dl_db is the derivative of the loss runction with respect to the scalar bias.
        # for the sigmoid function, it is the same as dl_dz
        
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        grad_w = np.round(dl_dw, 5)
        grad_b = round(float(dl_dz), 5)
        return (grad_w, grad_b)
