## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
import numpy as np

def mean(l): 
    return sum(l)/len(l)

def output(config,models, X, alpha):
    out = 0
    for i,model in enumerate(models):
        out += alpha[i]*model(X)
    return out 

def _pgd(config,model, X, y, epsilon, alpha, niters=100,step_size=0.01): 
    out = output(config,model,X, alpha)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    y_y=y.view(-1,1)
    for i in range(niters): 
        #opt = optim.Adam([X_pgd], lr=1e-3)
        #opt.zero_grad()
        #loss = nn.CrossEntropyLoss()(output(model,X_pgd, alpha), y)
        
        loss=-(torch.gather(output(config,model,X_pgd,alpha),1,y_y)).sum()
        #loss = output(model,X_pgd,alpha)
        #print(loss)
        loss.backward()
        eta = step_size*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        #X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd.data = X.data + eta
        X_pgd.data = torch.clamp(X_pgd.data, 0.0, 1.0)
    err_pgd = (output(config,model,X_pgd, alpha).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def pgd(config,loader, model, epsilon, alpha, niters=100, step_size=0.01, verbose=False,
        robust=False):
    return attack(config,loader, model, epsilon, alpha,verbose=verbose, atk=_pgd,
                  robust=robust)

def attack(config,loader, model, epsilon, alpha,verbose=False, atk=None,
           robust=False):
    # print(np.max(loader.dataset.data),np.min(loader.dataset.data))
    total_count = 0
    err_count = 0
    pgd_err_count = 0
    if verbose: 
        print("Requiring no gradients for parameters.")

    for i, (X,y) in enumerate(loader):
        X,y = Variable(X.cuda(), requires_grad=True), Variable(y.cuda().long())

        if y.dim() == 2: 
            y = y.squeeze(1)

        err, err_pgd = atk(config,model, X, y, epsilon, alpha)
        total_count += X.size(0)
        err_count += err
        pgd_err_count += err_pgd

    if verbose: 
        print('clean err: {} | PGD err: {}'.format(err, err_fgs))
    
    err_rate = err_count / total_count
    pgd_err_rate = pgd_err_count / total_count
    print('[TOTAL] clean err: {:6.4f} | PGD err: {:6.4f}'.format(err_rate, pgd_err_rate))
    #return total_err, total_fgs, total_robust
    return err_rate, pgd_err_rate

