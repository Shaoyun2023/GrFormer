import math
import torch
from torch import nn
from torch.autograd import Function
from args_fusion import args

dtype = torch.double
device = torch.device('cpu')


def calcuK(S):
    b, c, h = S.shape
    Sr = S.reshape(b, c, 1, h)
    Sc = S.reshape(b, c, h, 1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    K[torch.isnan(K)] = 0
    return K

def calcuK2(S):
    b, c,c2, h = S.shape
    Sr = S.reshape(b, c,c2, 1, h)
    Sc = S.reshape(b, c,c2, h, 1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    K[torch.isnan(K)] = 0
    return K


class FRMap(nn.Module):
    def __init__(self, input_size, output_size):
        super(FRMap, self).__init__()
        self.weight = nn.Parameter(torch.rand(input_size, output_size, dtype=torch.double) * 2 - 1.0)

    def forward(self, x):
        # weight, _ = torch.linalg.qr(self.weight)
        weight, _ = torch.qr(self.weight)

        output = torch.matmul(weight.transpose(-1, -2), x)

        return output


class QR(nn.Module):
    def __init__(self):
        super(QR, self).__init__()

    def forward(self, x):
        Q, R = torch.qr(x)
        # output = torch.matmul(Q,R)
        return Q

class QRComposition(nn.Module):
    def __init__(self):
        super(QRComposition, self).__init__()

    def forward(self, x):
        Q, R = torch.qr(x)

        # flipping                                                                                                              #torch.sign(torch.sign(...) + 0.5): 这一步似乎有点多余，因为torch.sign的输出加上0.5之后再次使用torch.sign似乎不会改变结果。「但其实是一个小技巧」，该操作的目的是确保0（即R的对角线上的零元素）被赋予正值+1。摘出这一部分逻辑来看：首先对R的对角元素进行符号判断，然后给结果加上0.5（这会让正数和零的结果都变成正数），最后再次进行符号判断，这确保了所有非负数（包括零）的符号都是+1。
        output = torch.matmul(Q, torch.diag_embed(torch.sign(torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        # output = torch.matmul(Q,R)
        return output


class Projmap(nn.Module):
    def __init__(self):
        super(Projmap, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.transpose(-1, -2))




class Orthmap(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return OrthmapFunction.apply(x, self.p)

class Orthmap2(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return OrthmapFunction2.apply(x, self.p)

class OrthmapFunction(Function):
    @staticmethod
    def forward(ctx, x, p):
        # x = x.to("cuda:0")
        # U, S, V = torch.linalg.svd(x)
        U, S, V = torch.svd(x)
        ctx.save_for_backward(U, S)
        res = U[..., :p]
        return res

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b, c, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(b, c, h, p)

        grad_output = torch.cat((grad_output, pad_zero), 3)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None

class OrthmapFunction2(Function):
    @staticmethod
    def forward(ctx, x, p):
        # U, S, V = torch.linalg.svd(x)
        U, S, V = torch.svd(x)
        ctx.save_for_backward(U, S)
        res = U[..., :p]
        return res

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b, c,c2, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(b, c,c2, h, p)

        grad_output = torch.cat((grad_output, pad_zero), 4)
        Ut = U.transpose(-1, -2)
        K = calcuK2(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None

class ProjPoolLayer_A(torch.autograd.Function):
    # AProjPooling  c/n ==0
    @staticmethod
    def forward(ctx, x, n=4):
        b, c, h, w = x.shape
        ctx.save_for_backward(n)
        new_c = int(math.ceil(c / n))
        new_x = [x[:, i:i + n].mean(1) for i in range(0, c, n)]
        return torch.cat(new_x, 1).reshape(b, new_c, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.saved_variables
        return torch.repeat_interleave(grad_output / n, n, 1)


# class ProjPoolLayer(nn.Module):
#     """ W-ProjPooling"""
#
#     def __init__(self, n=4):
#         super().__init__()
#         self.n = n
#
#     def forward(self, x):
#         avgpool = torch.nn.AvgPool2d(int(math.sqrt(self.n)))
#         # avgpool = torch.nn.MaxPool2d(int(math.sqrt(self.n)))
#         return avgpool(x)

class ProjPoolLayer(nn.Module):
    """ W-ProjPooling"""

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    def forward(self, x):
        avgpool = torch.nn.AvgPool2d(int(math.sqrt(self.n)))
        # avgpool = torch.nn.MaxPool2d(int(math.sqrt(self.n)))
        return avgpool(x)