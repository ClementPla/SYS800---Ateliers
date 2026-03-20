"""
Module: base class for all neural network layers.

Unlike the Operator-based approach, Modules here are thin wrappers
that compose Tensor operations. Because each Tensor operation already
records itself in the computational graph, there is NO need for
explicit backward() methods in Module — gradients flow automatically.
"""

import numpy as np
from neural_networks.params import Parameter


# ------------------------------------------------------------------
# Base Module
# ------------------------------------------------------------------
class Module:
    def parameters(self):
        """Recursively collect all Parameters in this module."""
        params = []
        for name, val in self.__dict__.items():
            if isinstance(val, Parameter):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ------------------------------------------------------------------
# Layers
# ------------------------------------------------------------------
class Linear(Module):
    """
    y = x @ W + b

    Because @ and + are Tensor operations that record themselves
    in the graph, we don't need to write any backward logic.
    """

    def __init__(self, in_features, out_features, bias=True):
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * scale,
            name="weight",
        )
        self.use_bias = bias
        if bias:
            self.bias = Parameter(np.zeros(out_features), name="bias")

    def forward(self, x):
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out

    def __repr__(self):
        in_f, out_f = self.weight.shape
        return f"Linear({in_f}, {out_f}, bias={self.use_bias})"


class Sequential(Module):
    """Apply a list of modules in order."""

    def __init__(self, *modules):
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

    def __repr__(self):
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(lines) + "\n)"


# ------------------------------------------------------------------
# Activation "modules" — thin wrappers around Tensor methods
# ------------------------------------------------------------------
class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    """
    Numerically stable softmax along axis=1 (class dimension).
    Built entirely from Tensor ops, so backward is automatic.
    """

    def forward(self, x):
        # Shift for numerical stability
        x_max = x.max(axis=1, keepdims=True)
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        return exp_x / exp_x.sum(axis=1, keepdims=True)