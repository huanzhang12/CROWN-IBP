## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import torch
import numpy as np
from torch.nn import Sequential, Conv2d, Linear, ReLU
from model_defs import Flatten, model_mlp_any
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BoundFlatten(torch.nn.Module):
    def __init__(self):
        super(BoundFlatten, self).__init__()

    def forward(self, x):
        self.shape = x.size()[1:]
        return x.view(x.size(0), -1)

    def interval_propagate(self, norm, h_U, h_L, eps):
        return norm, h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1), 0, 0, 0, 0

    def bound_backward(self, last_A):
        return last_A.view(last_A.size(0), last_A.size(1), *self.shape), 0

class BoundLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l

    def bound_backward(self, last_A):
        logger.debug('last_A %s', last_A.size())
        # propagate A to the next layer
        next_A = last_A.matmul(self.weight)
        logger.debug('next_A %s', next_A.size())
        # compute the bias of this layer
        sum_bias = last_A.matmul(self.bias)
        # print(sum_bias)
        logger.debug('sum_bias %s', sum_bias.size())
        return next_A, sum_bias

    def interval_propagate(self, norm, h_U, h_L, eps, C = None):
        # merge the specification
        if C is not None:
            # after multiplication with C, we have (batch, output_shape, prev_layer_shape)
            # we have batch dimension here because of each example has different C
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias

        if norm == np.inf:
            # Linf norm
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            if C is not None:
                center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
                deviation = weight_abs.matmul(diff.unsqueeze(-1))
                # these have an extra (1,) dimension as the last dimension
                center = center.squeeze(-1)
                deviation = deviation.squeeze(-1)
            else:
                # fused multiply-add
                center = torch.addmm(bias, mid, weight.t())
                deviation = diff.matmul(weight_abs.t())
        else:
            # L2 norm
            h = h_U # h_U = h_L, and eps is used
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if C is not None:
                center = weight.matmul(h.unsqueeze(-1)) + bias.unsqueeze(-1)
                center = center.squeeze(-1)
            else:
                center = torch.addmm(bias, h, weight.t())
            deviation = weight.norm(dual_norm, -1) * eps

        upper = center + deviation
        lower = center - deviation
        # output 
        return np.inf, upper, lower, 0, 0, 0, 0
            


class BoundConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BoundConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    @staticmethod
    def convert(l):
        nl = BoundConv2d(l.in_channels, l.out_channels, l.kernel_size, l.stride, l.padding, l.dilation, l.groups, l.bias is not None)
        nl.weight.data.copy_(l.weight.data)
        nl.bias.data.copy_(l.bias.data)
        logger.debug(nl.bias.size())
        logger.debug(nl.weight.size())
        return nl

    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        self.output_shape = output.size()[1:]
        return output

    def bound_backward(self, last_A):
        logger.debug('last_A %s', last_A.size())
        shape = last_A.size()
        # propagate A to the next layer, with batch concatenated together
        next_A = F.conv_transpose2d(last_A.view(shape[0] * shape[1], *shape[2:]), self.weight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
        logger.debug('next_A %s', next_A.size())
        logger.debug('bias %s', self.bias.size())
        # dot product
        # compute the bias of this layer, do a dot product
        sum_bias = (last_A.sum((3,4)) * self.bias).sum(2)
        # print(sum_bias)
        logger.debug('sum_bias %s', sum_bias.size())
        return next_A, sum_bias

    def interval_propagate(self, norm, h_U, h_L, eps):
        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = self.weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # L2 norm
            mid = h_U
            logger.debug('mid %s', mid.size())
            # TODO: consider padding here?
            deviation = torch.mul(self.weight, self.weight).sum((1,2,3)).sqrt() * eps
            logger.debug('weight %s', self.weight.size())
            logger.debug('deviation %s', deviation.size())
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            logger.debug('unsqueezed deviation %s', deviation.size())
        center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        logger.debug('center %s', center.size())
        upper = center + deviation
        lower = center - deviation
        return np.inf, upper, lower, 0, 0, 0, 0
    
class BoundReLU(ReLU):
    def __init__(self, prev_layer, inplace=False):
        super(BoundReLU, self).__init__(inplace)
        # ReLU needs the previous layer's bounds
        # self.prev_layer = prev_layer
    
    ## Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer):
        l = BoundReLU(prev_layer, act_layer.inplace)
        return l

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        # return F.relu(h_U), F.relu(h_L), -torch.tanh(1 + h_U * h_L).sum(), ((h_U > 0) & (h_L < 0)).sum().detach().cpu().item(), \
        # return F.relu(h_U), F.relu(h_L), 0, ((h_U > 0) & (h_L < 0)).sum().detach().cpu().item(), \
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        # stored upper and lower bounds will be used for backward bound propagation
        self.upper_u = h_U
        self.lower_l = h_L
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        return norm, F.relu(h_U), F.relu(h_L), tightness_loss, tightness_loss.detach().cpu().item(), \
               (h_U < 0).sum().detach().cpu().item(), (h_L > 0).sum().detach().cpu().item()

    def bound_backward(self, last_A):
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # ub_r = F.softplus(self.upper_u, beta=50)
        # lb_r = - F.softplus(self.lower_l, beta=50)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        upper_d = upper_d.unsqueeze(1)
        lower_d = (upper_d > 0.5).float()
        # lower_d = torch.sigmoid(100 * upper_d - 50) # differentiable version of lower slope (any slope between 0 and 1 is a valid bound)
        # Choose upper or lower bounds based on the sign of last_A
        neg_A = last_A.clamp(max=0)
        pos_A = last_A.clamp(min=0)
        next_A = upper_d * neg_A + lower_d * pos_A
        mult_A = neg_A.view(last_A.size(0), last_A.size(1), -1)
        sum_bias = mult_A.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        del self.upper_u
        del self.lower_l
        return next_A, sum_bias


class BoundSequential(Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model):
        layers = []
        for l in sequential_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1]))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten())
        return BoundSequential(*layers)


    ## High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        # start propagation from the last layer
        A, sum_b = list(self._modules.values())[-1].bound_backward(C)
        for i, module in enumerate(reversed(list(self._modules.values())[:-1])):
            logger.debug('before: %s', A.size())
            A, b = module.bound_backward(A)
            logger.debug('after: %s', A.size())
            sum_b += b
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug('Final A: %s', A.size())
        if norm == np.inf:
            x_U = x_U.view(x_U.size(0), -1, 1)
            x_L = x_L.view(x_U.size(0), -1, 1)
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0
            logger.debug('A_0 shape: %s', A.size())
            logger.debug('sum_b shape: %s', sum_b.size())
            # we only need the lower bound
            lb = A.bmm(center) - A.abs().bmm(diff)
            logger.debug('lb shape: %s', lb.size())
        else:
            x = x_U.view(x_U.size(0), -1, 1)
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            deviation = A.norm(dual_norm, -1) * eps
            lb = A.bmm(x) - deviation.unsqueeze(-1)
        lb = lb.squeeze(-1) + sum_b
        return lb, sum_b

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        losses = 0
        unstable = 0
        dead = 0
        alive = 0
        h_U = x_U
        h_L = x_L
        for i, module in enumerate(list(self._modules.values())[:-1]):
            # all internal layers should have Linf norm, except for the first layer
            norm, h_U, h_L, loss, uns, d, a = module.interval_propagate(norm, h_U, h_L, eps)
            # this is some stability loss used for initial experiments, not used in CROWN-IBP as it is not very effective
            losses += loss
            unstable += uns
            dead += d
            alive += a
        # last layer has C to merge
        norm, h_U, h_L, loss, uns, d, a = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, eps, C)
        losses += loss
        unstable += uns
        dead += d
        alive += a
        return h_U, h_L, losses, unstable, dead, alive

