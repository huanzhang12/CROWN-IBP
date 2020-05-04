## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##

# from convex_adversarial import Dense, DenseSequential

import torch
import torch.nn as nn

from model_defs import Flatten

def IBP_large(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


def IBP_debug(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential( 
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(), 
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(), 
        Flatten(),
        nn.Linear((in_dim//4) * (in_dim//4) * 1, 10), 
    )
    return model
