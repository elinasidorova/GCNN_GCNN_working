import torch
from torch import nn


class ConcatPooling(nn.Module):
    def __init__(self, input_dims):
        super(ConcatPooling, self).__init__()
        self.input_dims = input_dims
        self.output_dim = sum(input_dims)

    def forward(self, *tensors):
        if len(tensors) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} tensors, but {len(tensors)} were given")
        for i, tensor in enumerate(tensors):
            if tensor.shape[-1] != self.input_dims[i]:
                raise ValueError(
                    f"tensor {i} must have last dimension of {self.input_dims[i]}, but has {tensor.shape[-1]}")
        return torch.cat(tensors, dim=-1)


class SumPooling(nn.Module):
    def __init__(self, input_dims):
        super(SumPooling, self).__init__()
        self.output_dim = self.input_dims = input_dims

    def forward(self, *tensors):
        if len(tensors) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} tensors, but {len(tensors)} were given")
        for i, tensor in enumerate(tensors):
            if tensor.shape[-1] != self.input_dims:
                raise ValueError(
                    f"tensor {i} must have last dimension of {self.input_dims}, but has {tensor.shape[-1]}")
        return torch.sum(torch.cat([tensor.unsqueeze(0) for tensor in tensors], dim=0), dim=0)


class MaxPooling(nn.Module):
    def __init__(self, input_dims):
        super(MaxPooling, self).__init__()
        self.output_dim = self.input_dims = input_dims

    def forward(self, *tensors):
        if len(tensors) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} tensors, but {len(tensors)} were given")
        for i, tensor in enumerate(tensors):
            if tensor.shape[-1] != self.input_dims:
                raise ValueError(
                    f"tensor {i} must have last dimension of {self.input_dims}, but has {tensor.shape[-1]}")
        return torch.max(torch.cat([tensor.unsqueeze(0) for tensor in tensors], dim=0), dim=0)
