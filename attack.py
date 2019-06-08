## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import numpy as np
import torch
from argparser import argparser
from pgd import pgd
import os
import sys
from config import load_config, config_dataloader, config_modelloader


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)
    models, model_ids = config_modelloader(config, load_pretrain = True)
    models = [model.cuda() for model in models]
    # load dataset, depends on the dataset specified in config file
    batch_size = config["attack_params"]["batch_size"]
    train_loader, test_loader = config_dataloader(config, batch_size = batch_size, shuffle_train = False, normalize_input = False)

    eps_start = config["attack_params"]["eps_start"]
    eps_end = config["attack_params"]["eps_end"]
    eps_step = config["attack_params"]["eps_step"]
    for eps in np.linspace(eps_start, eps_end, eps_step):
        print('eps =', eps)
        """
        init = [1/len(models)]*len(models)
        init_t = torch.Tensor(init).cuda()
        print('naive on test')
        total_err, total_fgs = pgd(config,test_loader,models,eps, init_t)
        naive_test_error.append((total_err,total_fgs))
        print('naive on train')
        total_err, total_fgs = pgd(config,train_loader,models,eps, init_t)
        naive_train_error.append((total_err,total_fgs))
        """

        for i,model in enumerate(models):
            print('on '+model_ids[i])
            total_err, total_fgs = pgd(config,test_loader,[model],eps, [1])


