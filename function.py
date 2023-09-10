from torch.autograd import Function
import torch

class FakeQuantize(Function):
    
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize(x)
        x = qparam.dequantize(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
