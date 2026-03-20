"""
Parameter: a Tensor subclass that marks itself as a learnable parameter.

This distinction lets Module.parameters() collect only the
tensors that should be updated by an optimizer.
"""

from neural_networks.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, name=None):
        super().__init__(data, requires_grad=True)
        self.name = name

    def __repr__(self):
        return f"Parameter(name={self.name}, shape={self.shape})"