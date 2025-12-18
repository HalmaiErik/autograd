import torch
import core.node as n
import core.tensor as t
from core.utils import get_broadcast_dims


class Function(n.Node):
    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, *tensors: t.Tensor) -> list[t.Tensor]:
        raise NotImplementedError
    
    def backward(self, *output_grads: t.Tensor) -> list[t.Tensor]:
        raise NotImplementedError

    def apply(self, *inputs: t.Tensor) -> list[t.Tensor]:
        outputs = self.forward(*inputs)
        self.backFn = BackwardFunction(self.__class__.__name__ + 'Backward', self.backward, *inputs)
        
        for output in outputs:
            output.grad_fn = self.backFn
            output.add_next_node(self.backFn)
        
        return outputs


class BackwardFunction(n.Node):
    def __init__(self, name: str, backward, *tensors: t.Tensor):
        super().__init__(name)

        self.backward = backward
        for tensor in tensors:
            if tensor.is_leaf:
                gradAccumulator = GradientAccumulator(tensor)
                self.add_next_node(gradAccumulator)
            elif tensor.grad_fn:
                self.add_next_node(tensor.grad_fn)

    def apply(self, *output_grads: torch.Tensor):
        grads = self.backward(*output_grads)
        for i, node in enumerate(self.nextNodes):
            node.apply(grads[i])


class GradientAccumulator(n.Node):
    def __init__(self, tensor: torch.Tensor):
        super().__init__(self.__class__.__name__)
        self.tensor = tensor
        self.add_next_node(tensor)
    
    def apply(self, output_grad: torch.Tensor):
        self.tensor.grad = torch.zeros(self.tensor.data.shape) if self.tensor.grad == None else self.tensor.grad
        self.tensor.grad = self.tensor.grad + output_grad

        list_dims, list_not_keeps = get_broadcast_dims(self.tensor.data, self.tensor.grad)
        for dim in list_dims:
            if dim in list_not_keeps:
                self.tensor.grad = self.tensor.grad.sum(dim, keepdims=False)
            else:
                self.tensor.grad = self.tensor.grad.sum(dim, keepdims=True)


class Add(Function):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    
    def forward(self, *inputs: t.Tensor) -> list[t.Tensor]:
        self.t1 = inputs[0]
        self.t2 = inputs[1]

        output = t.Tensor(self.t1.data + self.t2.data, is_leaf=False)
        self.add_next_node(output)

        return [output]

    def backward(self, output_grad: torch.Tensor) -> list[torch.Tensor]:
        return [output_grad, output_grad]
    

class MatMul(Function):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    
    def forward(self, *inputs: t.Tensor) -> list[t.Tensor]:
        self.t1 = inputs[0]
        self.t2 = inputs[1]
        
        output = t.Tensor(self.t1.data @ self.t2.data, is_leaf=False)
        self.add_next_node(output)

        return [output]

    def backward(self, output_grad: torch.Tensor) -> list[torch.Tensor]:
        return [output_grad @ self.t2.data.T, self.t1.data.T @ output_grad]


class Tanh(Function):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    
    def forward(self, *inputs: t.Tensor) -> list[t.Tensor]:
        self.t = inputs[0]

        self.tanh = (torch.exp(2 * self.t.data) - 1) / (torch.exp(2 * self.t.data) + 1)
        output = t.Tensor(self.tanh, is_leaf=False)
        self.add_next_node(output)

        return [output]

    def backward(self, output_grad: torch.Tensor) -> list[torch.Tensor]:
        return [(1 - self.tanh ** 2) * output_grad]
        