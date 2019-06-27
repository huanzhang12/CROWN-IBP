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

    def interval_propagate(self, h_U, h_L):
        return h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1), 0, 0, 0, 0

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

    def interval_propagate(self, h_U, h_L, C = None):
        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0
        if C is not None:
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
            weight_abs = weight.abs()
            # after multiplication with C, we have (batch, output_shape, prev_layer_shape)
            # we have batch dimension here because of each example has different C
            center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
            deviation = weight_abs.matmul(diff.unsqueeze(-1))
            center = center.squeeze(-1)
            deviation = deviation.squeeze(-1)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias
            weight_abs = weight.abs()
            # fused multiply-add
            center = torch.addmm(bias, mid, weight.t())
            deviation = diff.matmul(weight_abs.t())
        upper = center + deviation
        lower = center - deviation
        return upper, lower, 0, 0, 0, 0


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

    def reshape(self, A):
        # TODO: avoid this permute!
        A = A.permute(0,4,1,2,3)
        return A.view(-1, *A.size()[2:])

    def extract(self, A, batch_size):
        # TODO: avoid this permute!
        A = A.view(batch_size, -1, *A.size()[1:])
        return A.permute(0,2,3,4,1)

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

    def interval_propagate(self, h_U, h_L):
        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0
        weight_abs = self.weight.abs()
        # bias has #filters parameters. Need to extend to (1, #out_C, #out_W, #out_H) (actually not necessary)
        # bias = self.bias.view(self.bias.size(0),1,1).expand(self.output_shape).unsqueeze(0)
        center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        upper = center + deviation
        lower = center - deviation
        return upper, lower, 0, 0, 0, 0
    
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

    def interval_propagate(self, h_U, h_L):
        # return F.relu(h_U), F.relu(h_L), -torch.tanh(1 + h_U * h_L).sum(), ((h_U > 0) & (h_L < 0)).sum().detach().cpu().item(), \
        # return F.relu(h_U), F.relu(h_L), 0, ((h_U > 0) & (h_L < 0)).sum().detach().cpu().item(), \
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        self.upper_u = h_U
        self.lower_l = h_L
        """
        h_U_unstab = h_U[self.unstab]
        h_L_unstab = h_L[self.unstab]
        tightness_loss = (-h_U_unstab * h_L_unstab / (h_U_unstab - h_L_unstab)).sum()
        """
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        return F.relu(h_U), F.relu(h_L), tightness_loss, tightness_loss.detach().cpu().item(), \
               (h_U < 0).sum().detach().cpu().item(), (h_L > 0).sum().detach().cpu().item()
        # return F.relu(h_U), F.relu(h_L), ((h_U > 0) & (h_L < 0)).sum()
        # return F.relu(h_U), F.relu(h_L), h_U.numel()

    def bound_backward(self, last_A):
        logger.debug('last_A %s', last_A.size())
        guard_eps = 1e-5
        # unstable neurons
        lb = self.lower_l
        ub = self.upper_u
        unstab = self.unstab
        lb_unstab = lb[unstab]
        ub_unstab = ub[unstab]
        """
        upper_d = torch.zeros_like(lb)
        lower_d = torch.zeros_like(lb)
        logger.debug('upper_d %s', upper_d.size())
        # active neurons
        active = (lb > guard_eps).detach()
        upper_d[active] = 1.0
        lower_d[active] = 1.0
        # inactive neurons (default)
        # CROWN bound: upper bound
        upper_d[unstab] = ub_unstab / (ub_unstab - lb_unstab)
        unstab_ularge = (unstab & (ub > -lb)).detach() # a guard epsilon is added to avoid division by 0
        # CROWN bound: choose 1.0 as the lower bound for u > |l|; otherwise 0.0
        lower_d[unstab_ularge] = 1.0
        # CROWN bound: for positive element in A 
        next_A = upper_d.unsqueeze(1) * last_A.clamp(max=0) + lower_d.unsqueeze(1) * last_A.clamp(min=0)
        """
        ub_slope = ub_unstab / (ub_unstab - lb_unstab)
        # copy and fill inactive elements with 0, other elements are multiplied by 1 automatically
        # for active neurons, we don't need to do anything. Inactive neurons are filled with 0
        inactive = (ub < -guard_eps).detach()
        next_A = last_A.masked_fill(inactive.unsqueeze(1), 0.0)
        # indices of all unstable neurons
        unstab_s = unstab.view(self.unstab.size(0), -1).nonzero()
        # now fill in values for all unstable neurons.
        # we first transpose last_A to (output_shape, batch, *layer_size)
        # those are unstable elements
        unstable_A = torch.transpose(last_A,0,1).masked_select(unstab.unsqueeze(0))
        # now, we get the A's element for all unstable neurons, shape (output_shape, all unstable neurons in batches)
        unstable_A = unstable_A.view(last_A.size(1), -1)
        # for element in A < 0 we choose u/(u-l)
        A_neg_slopes = unstable_A.clamp(max=0) * ub_slope
        # for element in A > 0, we choose the flexible lower bound
        # for the unstable neurons with |LB| < |UB| we will choose 1 as the lower slope
        lb_slope = torch.zeros_like(ub_slope)
        lb_slope[ub_unstab > -lb_unstab] = 1.0
        A_pos_slopes = unstable_A.clamp(min=0) * lb_slope
        # now form the slopes for all unstable neurons
        unstable_slopes = A_pos_slopes + A_neg_slopes
        # and fill them into the unstable neuron places, similiar to the way we get them
        next_A = torch.transpose(next_A,0,1).masked_scatter_(unstab.unsqueeze(0), unstable_slopes)
        # swap back to the original order
        next_A = torch.transpose(next_A,0,1)

        logger.debug('next_A %s', next_A.size())
        # choose the bias term according to next_A
        mult_A = next_A.view(last_A.size(0), last_A.size(1), -1)
        logger.debug('next_A reshape %s', mult_A.size())
        llb = torch.zeros_like(lb)
        llb[unstab] = lb_unstab # llb is actually the intercept of UPPER bound
        sum_bias = mult_A.clamp(max=0).matmul(-llb.view(llb.size(0), -1, 1)).squeeze(-1)
        # print('relu', sum_bias)
        logger.debug('sum_bias %s', sum_bias.size())
        # done, delete saved bounds
        del self.upper_u
        del self.lower_l
        del self.unstab
        return next_A, sum_bias

    def forward(self, input):
        output = super(BoundReLU, self).forward(input)
        # give an initial bound such that even without bound propagation, a global Lipschitz constant can be obtained
        # self.upper_u = torch.ones_like(input).unsqueeze(-1)
        # self.upper_u *= np.inf
        # self.lower_l = torch.ones_like(input).unsqueeze(-1)
        # self.lower_l *= -np.inf
        return output

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
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
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
        x_U = x_U.view(x_U.size(0), -1, 1)
        x_L = x_L.view(x_U.size(0), -1, 1)
        center = (x_U + x_L) / 2.0
        diff = (x_U - x_L) / 2.0
        logger.debug('A_0 shape: %s', A.size())
        logger.debug('sum_b shape: %s', sum_b.size())
        # we only need the lower bound
        lb = A.bmm(center) - A.abs().bmm(diff)
        logger.debug('lb shape: %s', lb.size())
        lb = lb.squeeze(-1) + sum_b
        return lb, sum_b

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        h_U = x_U
        h_L = x_L
        losses = 0
        unstable = 0
        dead = 0
        alive = 0
        for i, module in enumerate(list(self._modules.values())[:-1]):
            h_U, h_L, loss, uns, d, a = module.interval_propagate(h_U, h_L)
            losses += loss
            unstable += uns
            dead += d
            alive += a
        # last layer has C to merge
        h_U, h_L, loss, uns, d, a = list(self._modules.values())[-1].interval_propagate(h_U, h_L, C)
        losses += loss
        unstable += uns
        dead += d
        alive += a
        return h_U, h_L, losses, unstable, dead, alive

