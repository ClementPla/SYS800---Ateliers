
import numpy as np
from autograd.tensor import Tensor


class Context:
    """
    Stores information needed for the backward pass.
    Mimics PyTorch's ctx object.
    """

    def __init__(self):
        self.saved_tensors = ()
        self._saved_data = {}

    def save_for_backward(self, *tensors):
        """Save tensors needed for gradient computation."""
        self.saved_tensors = tensors

    def save(self, **kwargs):
        """Save arbitrary data (shapes, masks, constants, etc.)."""
        self._saved_data.update(kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._saved_data[name]
        except KeyError:
            raise AttributeError(
                f"Nothing saved under '{name}'. "
                f"Did you call ctx.save({name}=...) in forward?"
            )


class Function:
    """
    Base class for custom autograd operations.

    Subclasses must implement:
        forward(ctx, *inputs)  -> Tensor (the result)
        backward(ctx, grad_output) -> tuple of gradients (one per input)
    """

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        """
        Run forward, then wire the result into the computational graph
        so that .backward() will call our custom backward.
        """
        ctx = Context()

        # Run the custom forward
        output = cls.forward(ctx, *inputs)

        # Make sure output is a Tensor
        if not (hasattr(output, "data") and hasattr(output, "grad")):
            output = Tensor(output)

        # Wire into the graph: the output's _prev are the input tensors
        tensor_inputs = [inp for inp in inputs if hasattr(inp, "data") and hasattr(inp, "grad")]
        output._prev = set(tensor_inputs)
        output._op = cls.__name__

        # The key: override _backward to call our custom backward
        def _backward():
            grads = cls.backward(ctx, output.grad)

            # backward returns a tuple of grads, one per Tensor input
            if not isinstance(grads, tuple):
                grads = (grads,)

            idx = 0
            for inp in inputs:
                if hasattr(inp, "data") and hasattr(inp, "grad"):
                    if idx < len(grads) and grads[idx] is not None:
                        grad = grads[idx]
                        # Allow returning numpy arrays or Tensors
                        if hasattr(grad, "data") and hasattr(grad, "grad"):
                            grad = grad.data
                        inp.grad += grad
                    idx += 1

        output._backward = _backward
        return output