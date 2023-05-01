import torch


def mean_unifunc(x1, x2): return (x1 + x2) / 2


def max_unifunc(x1, x2): return torch.max(x1, x2)


def concat_unifunc(x1, x2): return torch.cat((x1, x2), dim=-1)
