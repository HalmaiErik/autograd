import torch

import core.functions as f
import core.node as n
from core.utils import get_broadcast_dims


class Tensor(n.Node):
    def __init__(self, data: torch.Tensor, is_leaf=True, name: str=None):
        super().__init__(Tensor.__name__, given_name=name)

        self.data = data
        self.grad: torch.Tensor = None
        self.grad_fn: f.BackwardFunction = None
        self.shape = self.data.shape
        self.dim = self.data.dim
        self.is_leaf = is_leaf

    def backwards(self, in_grad: torch.Tensor):
        self.grad = in_grad.expand(self.data.shape)

        list_dims, list_not_keeps = get_broadcast_dims(self.grad, self.data)
        for dim in list_dims:
            if dim in list_not_keeps:
                self.grad = self.grad.sum(dim, keepdims=False)
            else:
                self.grad = self.grad.sum(dim, keepdims=True)

        self.grad_fn.apply(self.grad)

    def zero_grad(self):
        self.grad = torch.zeros(self.data.shape)
    
    def __add__(self, other: Tensor) -> Tensor:
        func = f.Add()
        output_list = func.apply(self, other)

        self.add_next_node(func)
        other.add_next_node(func)

        return output_list[0]

    def __matmul__(self, other: Tensor) -> Tensor:
        func = f.MatMul()
        output_list = func.apply(self, other)

        self.add_next_node(func)
        other.add_next_node(func)

        return output_list[0]

    def tanh(self):
        func = f.Tanh()
        output_list = func.apply(self)

        self.add_next_node(func)

        return output_list[0]

    def __repr__(self):
        return super().__repr__() + "\n-------------\n" + str(self.data) + "\n-------------\n" + str(self.grad)
