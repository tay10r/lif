import torch

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        q = torch.round(x * 255)
        q = torch.clamp(q, 0, 255)
        return q * (1.0 / 255.0)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs