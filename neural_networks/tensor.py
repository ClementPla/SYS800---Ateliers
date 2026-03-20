"""
Tensor: the core data structure for automatic differentiation.

Each Tensor wraps a numpy array and optionally tracks:
- which operation created it (_op)
- which tensors were its inputs (_children)
- a local backward function (_backward) that computes gradients

When .backward() is called on a scalar tensor (e.g. a loss),
it performs a reverse topological traversal of the computational
graph, calling each node's _backward to accumulate gradients.
"""

import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        # Duck-type: if it looks like a Tensor/Parameter, unwrap it.
        # Using hasattr instead of isinstance avoids cross-module import issues.
        while hasattr(data, "data") and not isinstance(data, np.ndarray):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad

        # Computational graph bookkeeping
        self._backward = lambda: None  # no-op by default
        self._prev = set(_children)
        self._op = _op  # label for debugging / visualization

    # ------------------------------------------------------------------
    # Topological-sort backward pass
    # ------------------------------------------------------------------
    def backward(self):
        """Backpropagate from this tensor through the entire graph."""
        # Build topological order
        topo = []
        visited = set()

        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Seed gradient: d(loss)/d(loss) = 1
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # Reverse order: from output back to inputs
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Arithmetic operators — each one builds a graph edge
    # ------------------------------------------------------------------
    @staticmethod
    def _as_tensor(obj):
        """Ensure obj is a Tensor. Uses duck-typing to avoid cross-module isinstance issues."""
        if hasattr(obj, "data") and hasattr(obj, "grad"):
            return obj  # Already a Tensor (or Parameter, etc.)
        return Tensor(obj, requires_grad=False)

    def __add__(self, other):
        other = Tensor._as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # Handle broadcasting: sum over axes that were broadcast
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = Tensor._as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = Tensor._as_tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(
                out.grad * (-self.data / (other.data ** 2)), other.data.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = Tensor._as_tensor(other)
        return other / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only scalar exponents supported"
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = Tensor._as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                # Restore the reduced dimension for broadcasting
                grad = np.expand_dims(grad, axis=axis)
            self.grad += grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), "mean")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += grad * np.ones_like(self.data) / n

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), "max")

        def _backward():
            if axis is None:
                mask = (self.data == np.max(self.data)).astype(np.float64)
                # Split gradient among tied values
                mask /= mask.sum()
                self.grad += out.grad * mask
            else:
                max_vals = np.max(self.data, axis=axis, keepdims=True)
                mask = (self.data == max_vals).astype(np.float64)
                mask /= mask.sum(axis=axis, keepdims=True)
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += grad * mask

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Unary / elementwise operations
    # ------------------------------------------------------------------
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^a)/da = e^a
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            # d(ln a)/da = 1/a
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0).astype(np.float64)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            # d(σ)/da = σ(1-σ)
            self.grad += out.grad * out.data * (1 - out.data)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            # d(tanh)/da = 1 - tanh^2
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            full_grad = np.zeros_like(self.data)
            full_grad[idx] = out.grad
            self.grad += full_grad

        out._backward = _backward
        return out


# ------------------------------------------------------------------
# Helper: handle broadcasting in backward pass
# ------------------------------------------------------------------
def _unbroadcast(grad, target_shape):
    """
    When a + b involves broadcasting (e.g. (3,4) + (4,)),
    the gradient has the broadcast shape and must be summed
    over the axes that were added/expanded.
    """
    # Add leading dimensions if grad has more dims
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum over axes where target_shape is 1 but grad is larger
    for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad