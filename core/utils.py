import torch

def get_broadcast_dims(input: torch.Tensor, output: torch.Tensor):
    list_dims = []
    list_not_keeps = []

    if input.dim() < output.dim():
        table = torch.zeros(input.dim(), output.dim())
        for i, v_i in enumerate(input.size()):
            for j, v_j in enumerate(output.size()):
                if v_i == v_j and all(table[i, :j] == 0):  # just accept one-to-one mapping
                    table[i, j] = 1

        for k in range(output.dim()):
            if all(table[:, k] == 0):  # add dimension here
                torch.unsqueeze(input, k)
                list_not_keeps.append(k)

    for i, (l1, l2) in enumerate(zip(input.size(), output.size())):
        if l1 < l2:
            list_dims.append(i)

    return list_dims, set(list_not_keeps)