import torch

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        torch.abs
        q = torch.round(x * 255)
        q = torch.clamp(q, 0, 255)
        return q * (1.0 / 255.0)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class STEQuantizePerVectorAbsMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
        reduce_dims = tuple(range(1, x.ndim))
        s = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)

        x_scaled = torch.clamp(x / s, -1.0, 1.0)

        q = torch.round(x_scaled * 127.0)
        q = torch.clamp(q, -127.0, 127.0)

        x_hat = (q / 127.0) * s
        return x_hat, s

    @staticmethod
    def backward(ctx, grad_x_hat: torch.Tensor, grad_s: torch.Tensor):
        return grad_x_hat, None
