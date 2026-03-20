"""
Loss functions — built entirely from Tensor operations.

Because everything is composed of graph-aware Tensor ops,
calling loss.backward() automatically computes all gradients.
"""

import numpy as np
from neural_networks.tensor import Tensor
from neural_networks.nn import Module


class MSELoss(Module):
    """Mean Squared Error: L = mean((y_pred - y_true)^2)"""

    def forward(self, y_pred, y_true):
        if not (hasattr(y_true, "data") and hasattr(y_true, "grad")):
            y_true = Tensor(y_true, requires_grad=False)
        diff = y_pred - y_true
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """
    Cross-entropy with built-in log-softmax for numerical stability.

    Expects:
        logits: (batch_size, num_classes) raw scores
        targets: (batch_size,) integer class labels
    """

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # Log-softmax (numerically stable)
        x_max = logits.max(axis=1, keepdims=True)
        shifted = logits - x_max
        log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
        log_probs = shifted - log_sum_exp

        # One-hot encode targets, then pick the log-prob of the correct class
        if hasattr(targets, "data") and hasattr(targets, "grad"):
            targets_np = targets.data.astype(int)
        else:
            targets_np = np.array(targets, dtype=int)

        one_hot = np.zeros((batch_size, num_classes), dtype=np.float64)
        one_hot[np.arange(batch_size), targets_np] = 1.0
        one_hot_tensor = Tensor(one_hot, requires_grad=False)

        # -sum(one_hot * log_probs) / batch_size
        loss = -(one_hot_tensor * log_probs).sum() / batch_size
        return loss