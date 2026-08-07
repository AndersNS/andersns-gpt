import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        xa = np.asarray(x, dtype=np.float64)
        W1a = np.asarray(W1, dtype=np.float64)
        b1a = np.asarray(b1, dtype=np.float64)
        W2a = np.asarray(W2, dtype=np.float64)
        b2a = np.asarray(b2, dtype=np.float64)
        y_truea = np.asarray(y_true, dtype=np.float64)

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)

        # forwards pass
        z1 = W1a @ xa + b1a # weights 
        a1 = np.maximum(0, z1) # ReLU

        predictions = W2a @ a1 + b2a

        error = predictions - y_truea
        # MSE (mean squared error)
        loss = np.mean(error * error)

        # backwards
        # chain rule
        # MSE loss with respect to predictions:
        d_pred = (2 / y_truea.size) * error

        dW2 = np.outer(d_pred, a1)
        db2 = d_pred

        # backprop through ReLU layer
        da1 = W2a.T @ d_pred
        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, xa)
        db1 = dz1

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }

