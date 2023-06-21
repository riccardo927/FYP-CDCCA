import torch
import torch.nn as nn
from torch.autograd import gradcheck
import math


# define backpropagation using CR-calculus
class WidelyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, weight2, bias):
        ctx.save_for_backward(input, weight1, weight2, bias)
        z = torch.mm(input, weight1.t()) + torch.mm(input.conj(), weight2.t()) + bias
        return z

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, weight2, bias = ctx.saved_tensors
        grad_input = grad_weight1 = grad_weight2 = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = (grad_output.conj().mm(weight1) + grad_output.mm(weight2.conj())).conj()
        if ctx.needs_input_grad[1]:
            grad_weight1 = grad_output.t().mm(input.conj())
        if ctx.needs_input_grad[2]:
            grad_weight2 = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight1, grad_weight2, grad_bias


class WidelyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(WidelyLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.complex(torch.Tensor(out_features, in_features),
                                                  torch.Tensor(out_features, in_features)))
        self.weight2 = nn.Parameter(torch.complex(torch.Tensor(out_features, in_features),
                                                  torch.Tensor(out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.complex(torch.Tensor(out_features),
                                                   torch.Tensor(out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1.real, a=math.sqrt(5))#, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.weight1.imag, a=math.sqrt(5))#, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.weight2.real, a=math.sqrt(5))#, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.weight2.imag, a=math.sqrt(5))#, nonlinearity='leaky_relu')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1.real)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias.real, -bound, bound)
            nn.init.uniform_(self.bias.imag, -bound, bound)

    def forward(self, input):
        # z = torch.mm(input, self.weight1.t()) + torch.mm(input.conj(), self.weight2.t()) + self.bias
        # return z
        return WidelyLinearFunction.apply(input, self.weight1, self.weight2, self.bias)

class ComplexArctan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        output = torch.atan(z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        grad_input = (grad_output.conj() / (1 + torch.square(z)))
        return grad_input.conj()

class ComplexSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        output = torch.sigmoid(z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        grad_input = grad_output.conj() * torch.sigmoid(z) * (1 - torch.sigmoid(z))
        return grad_input.conj()


class ComplexActivation(nn.Module):
    def forward(self, input):
        # return torch.atan(input)
        return ComplexArctan.apply(input)
        #return ComplexSigmoid.apply(input)

class ComplexRelu(nn.Module):
    def forward(self, input):
        f = nn.ReLU()
        z = torch.view_as_real(input)
        f_z = f(z)
        return torch.view_as_complex(f_z)

# class ComplexBatchNorm1d(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
#         super(ComplexBatchNorm1d, self).__init__()
#         self.bn_r = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
#         self.bn_i = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

#     def forward(self, input):
#         output = self.bn_r(input.real) + 1j*self.bn_i(input.imag)
#         return output

# check gradients bt running these tests singularly

# test 1: widely linear function
# widely = WidelyLinearFunction.apply
# input = (torch.randn(100,1,dtype=torch.cdouble,requires_grad=True), torch.randn(128,1,dtype=torch.cdouble,requires_grad=True), torch.randn(128,1,dtype=torch.cdouble,requires_grad=True), torch.randn(128,dtype=torch.cdouble,requires_grad=True))
# test = gradcheck(widely, input, eps=1e-6, atol=1e-4)
# print(test)

# test 2: activation test
# change activation to test
# activation = ComplexSigmoid.apply #ComplexArctan.apply
# input = torch.randn(100,1,dtype=torch.cdouble,requires_grad=True)
# test = gradcheck(activation, input, eps=1e-6, atol=1e-4)
# print(test)