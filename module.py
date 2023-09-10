import torch.nn as nn
import torch
from function import *
import torch.nn.functional as F


def calcScaleZeroPoint(min, max, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1
    scale = (max - min) / (qmax - qmin)
    zero_point = qmax - max / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min.device)
    elif zero_point > qmax:
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max.device)

    zero_point.round_()
    return scale, zero_point


def quantize(tensor, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1

    x = zero_point + tensor / scale
    x.clamp_(qmin, qmax).round_()

    return x


def dequantize(tensor, scale, zero_point):
    x = (tensor - zero_point) * scale
    return x


class QParam(nn.Module):
    def __init__(self, num_bits=8):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("min", min)
        self.register_buffer("max", max)

    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        if self.min.nelement() == 0 or self.min.data < tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        self.scale, self.zero_point = calcScaleZeroPoint(
            self.min, self.max, self.num_bits)

    def quantize(self, tensor):
        return quantize(tensor, self.scale, self.zero_point, self.num_bits)

    def dequantize(self, tensor):
        return dequantize(tensor, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass


class QModule(nn.Module):
    def __init__(self, qi=True,  qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self):
        raise NotImplementedError("quantize_inference should be implemented")


class QConv2d(QModule):
    def __init__(self, conv_moduel, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_moduel
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, please provide')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, please provide')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data
        self.conv_module.weight.data = self.qw.quantize(
            self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize(
            self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale, zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)

        x = F.conv2d(x, FakeQuantize.apply(self.conv_module.weight, self.qw), self.conv_module.bias, stride=self.conv_module.stride,
                     padding=self.conv_module.padding,  dilation=self.conv_module.dilation, groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = x * self.M
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x


class QLinear(QModule):
    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=8)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))

    def freeze(self, qi = None, qo = None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, please provide')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, please provide')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data
        self.fc_module.weight.data = self.qw.quantize(
            self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = quantize(
            self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale, zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x
    
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    

class QRelu(QModule):
    def __init__(self, qi=False, num_bits=None):
        super(QRelu, self).__init__(qi = qi , num_bits=num_bits)
        
    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, please provide')
        
        if qi is not None:
            self.qi = qi
            
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        x = F.relu(x)
        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x
    

class QMaxPooling2d(QModule):
    def __init__(self, kernel_size = 3, stride = 1, padding = 0, qi = False, num_bits = None):
        super(QMaxPooling2d, self).__init__(qi = qi , num_bits= num_bits)
        self.kernel_size = kernel_size
        self.stride =stride
        self.padding = padding
        
    def freeze(self, qi = None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, please provide')
        
        if qi is not None:
            self.qi = qi
            
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x
    
    def quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
