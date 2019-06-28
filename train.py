## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from model_defs import Flatten, model_mlp_any, model_cnn_1layer, model_cnn_2layer, model_cnn_4layer, model_cnn_3layer
from bound_layers import BoundSequential, BoundLinear, BoundConv2d
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
from convex_adversarial import DualNetwork

from config import load_config, get_path, config_modelloader, config_dataloader, update_dict
from argparser import argparser
# sys.settrace(gpu_profile)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()


def Train(model, t, loader, start_eps, end_eps, max_eps, norm, logger, verbose, train, opt, method, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    num_class = 10
    losses = AverageMeter()
    l1_losses = AverageMeter()
    errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    relu_activities = AverageMeter()
    bound_bias = AverageMeter()
    bound_diff = AverageMeter()
    unstable_neurons = AverageMeter()
    dead_neurons = AverageMeter()
    alive_neurons = AverageMeter()
    batch_time = AverageMeter()
    # initial 
    kappa = 1
    factor = 1
    if train:
        model.train()
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa)
    total = len(loader.dataset)
    batch_size = loader.batch_size
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    batch_eps = np.linspace(start_eps, end_eps, (total // batch_size) + 1)
    model_range = 0.0
    if end_eps < 1e-6:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
    for i, (data, labels) in enumerate(loader): 
        start = time.time()
        eps = batch_eps[i]
        if train:   
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0),num_class-1,num_class))
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)

        # FIXME: Assume data is from range 0 - 1
        if kwargs["bounded_input"]:
            assert loader.std == [1,1,1] or loader.std == [1]
            # bounded input only makes sense for Linf perturbation
            assert norm == np.inf
            data_ub = (data + eps).clamp(max=1.0)
            data_lb = (data - eps).clamp(min=0.0)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps

        # omit the regular cross entropy, since we use robust error
        output = model(data)
        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), data.size(0))
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()
        
        """
        print('prediction:  ', output)
        ub, lb, _, _, _, _ = model.interval_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
        lb = lb_s.scatter(1, sa_labels, lb)
        print('interval ub: ', ub)
        print('interval lb: ', lb)
        lb, _ = model.backward_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
        lb = lb_s.scatter(1, sa_labels, lb)
        print('full lb: ', lb)
        input()
        """

        if verbose or method != "natural":
            if kwargs["bound_type"] == "convex-adv":
                # Wong and Kolter's bound, or equivalently Fast-Lin
                if kwargs["convex-proj"] is not None:
                    proj = kwargs["convex-proj"]
                    if norm == np.inf:
                        norm_type = "l1_median"
                    elif norm == 2:
                        norm_type = "l2_normal"
                    else:
                        raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
                else:
                    proj = None
                    if norm == np.inf:
                        norm_type = "l1"
                    elif norm == 2:
                        norm_type = "l2"
                    else:
                        raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
                if loader.std == [1] or loader.std == [1, 1, 1]:
                    convex_eps = eps
                else:
                    convex_eps = eps / np.mean(loader.std)
                    # for CIFAR we are roughly / 0.2
                    # FIXME this is due to a bug in convex_adversarial, we cannot use per-channel eps
                if norm == np.inf:
                    # bounded input is only for Linf
                    if kwargs["bounded_input"]:
                        # FIXME the bounded projection in convex_adversarial has a bug, data range must be positive
                        data_l = 0.0
                        data_u = 1.0
                    else:
                        data_l = -np.inf
                        data_u = np.inf
                else:
                    data_l = data_u = None
                f = DualNetwork(model, data, convex_eps, proj = proj, norm_type = norm_type, bounded_input = kwargs["bounded_input"], data_l = data_l, data_u = data_u)
                lb = f(c)
            elif kwargs["bound_type"] == "interval":
                ub, lb, relu_activity, unstable, dead, alive = model.interval_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
            elif kwargs["bound_type"] == "crown-interval":
                ub, ilb, relu_activity, unstable, dead, alive = model.interval_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
                crown_final_factor = kwargs['final-beta']
                factor = (max_eps - eps * (1.0 - crown_final_factor)) / max_eps
                if factor < 1e-5:
                    lb = ilb
                else:
                    if kwargs["runnerup_only"]:
                        # regenerate a smaller c, with just the runner-up prediction
                        # mask ground truthlabel output, select the second largest class
                        # print(output)
                        # torch.set_printoptions(threshold=5000)
                        masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                        # print(masked_output)
                        # location of the runner up prediction
                        runner_up = masked_output.max(1)[1]
                        # print(runner_up)
                        # print(labels)
                        # get margin from the groud-truth to runner-up only
                        runnerup_c = torch.eye(num_class).type_as(data)[labels]
                        # print(runnerup_c)
                        # set the runner up location to -
                        runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                        runnerup_c = runnerup_c.unsqueeze(1).detach()
                        # print(runnerup_c)
                        # get the bound for runnerup_c
                        clb, bias = model.backward_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
                        clb = clb.expand(clb.size(0), num_class - 1)
                    else:
                        # get the CROWN bound using interval bounds
                        clb, bias = model.backward_range(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c)
                        bound_bias.update(bias.sum() / data.size(0))
                    # how much better is crown-ibp better than ibp?
                    diff = (clb - ilb).sum().item()
                    bound_diff.update(diff / data.size(0), data.size(0))
                    # lb = torch.max(lb, clb)
                    lb = clb * factor + ilb * (1 - factor)
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"])

            lb = lb_s.scatter(1, sa_labels, lb)
            robust_ce = CrossEntropyLoss()(-lb, labels)
            if kwargs["bound_type"] != "convex-adv":
                relu_activities.update(relu_activity.detach().cpu().item() / data.size(0), data.size(0))
                unstable_neurons.update(unstable / data.size(0), data.size(0))
                dead_neurons.update(dead / data.size(0), data.size(0))
                alive_neurons.update(alive / data.size(0), data.size(0))

        if method == "robust":
            loss = robust_ce
        elif method == "robust_activity":
            loss = robust_ce + kwargs["activity_reg"] * relu_activity
        elif method == "natural":
            loss = regular_ce
        elif method == "robust_natural":
            natural_final_factor = kwargs["final-kappa"]
            kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            loss = (1-kappa) * robust_ce + kappa * regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if "l1_reg" in kwargs:
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + (reg * torch.sum(torch.abs(param)))
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))
        if train:
            loss.backward()
            opt.step()

        batch_time.update(time.time() - start)
        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            # robust_ce_losses.update(robust_ce, data.size(0))
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))
        if i % 50 == 0 and train:
            logger.log(  '[{:2d}:{:4d}]: eps {:4f}  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
                    'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                    'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                    'Err {errors.val:.4f} ({errors.avg:.4f})  '
                    'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
                    'Uns {unstable.val:.1f} ({unstable.avg:.1f})  '
                    'Dead {dead.val:.1f} ({dead.avg:.1f})  '
                    'Alive {alive.val:.1f} ({alive.avg:.1f})  '
                    'Tightness {tight.val:.5f} ({tight.avg:.5f})  '
                    'Bias {bias.val:.5f} ({bias.avg:.5f})  '
                    'Diff {diff.val:.5f} ({diff.avg:.5f})  '
                    'R {model_range:.3f}  '
                    'beta {factor:.3f} ({factor:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
                    unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
                    tight = relu_activities, bias = bound_bias, diff = bound_diff,
                    model_range = model_range, 
                    factor=factor, kappa = kappa))
    
                    
    logger.log(  '[FINAL RESULT epoch:{:2d} eps:{:.4f}]: '
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
        'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
        'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
        'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
        'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
        'Uns {unstable.val:.3f} ({unstable.avg:.3f})  '
        'Dead {dead.val:.1f} ({dead.avg:.1f})  '
        'Alive {alive.val:.1f} ({alive.avg:.1f})  '
        'Tight {tight.val:.5f} ({tight.avg:.5f})  '
        'Bias {bias.val:.5f} ({bias.avg:.5f})  '
        'Diff {diff.val:.5f} ({diff.avg:.5f})  '
        'Err {errors.val:.4f} ({errors.avg:.4f})  '
        'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
        'R {model_range:.3f}  '
        'beta {factor:.3f} ({factor:.3f})  '
        'kappa {kappa:.3f} ({kappa:.3f})  \n'.format(
        t, eps, batch_time=batch_time,
        loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
        regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
        unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
        tight = relu_activities, bias = bound_bias, diff = bound_diff,
        model_range = model_range, 
        kappa = kappa, factor=factor))
    for i, l in enumerate(model):
        if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
            norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
            logger.log('layer {} norm {}'.format(i, norm))
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg

def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader(config)

    converted_models = [BoundSequential.convert(model) for model in models]

    for model, model_id, model_config in zip(converted_models, model_names, config["models"]):
        model = model.cuda()

        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])

        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_factor = train_config["lr_decay_factor"]
        # parameters specific to a training method
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])

        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")

        eps_schedule = [0] * schedule_start + list(np.linspace(starting_epsilon, end_epsilon, schedule_length))
        max_eps = end_epsilon
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
        model_name = get_path(config, model_id, "model", load = False)
        best_model_name = get_path(config, model_id, "best_model", load = False)
        print(model_name)
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0
        for t in range(epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if t >= len(eps_schedule):
                eps = end_epsilon
            else:
                epoch_start_eps = eps_schedule[t]
                if t + 1 >= len(eps_schedule):
                    epoch_end_eps = epoch_start_eps
                else:
                    epoch_end_eps = eps_schedule[t+1]
            
            logger.log("Epoch {}, learning rate {}, epsilon {:.6f} - {:.6f}".format(t, lr_scheduler.get_lr(), epoch_start_eps, epoch_end_eps))
            # with torch.autograd.detect_anomaly():
            start_time = time.time()
            Train(model, t, train_data, epoch_start_eps, epoch_end_eps, max_eps, norm, logger, verbose, True, opt, method, **method_param)
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            with torch.no_grad():
                # evaluate
                err, clean_err = Train(model, t, test_data, epoch_end_eps, epoch_end_eps, max_eps, norm, logger, verbose, False, None, method, **method_param)

            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.state_dict(), 
                    'epoch' : t,
                    }, model_name)

            # save the best model after we reached the schedule
            if t >= len(eps_schedule):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                            'state_dict' : model.state_dict(), 
                            'robust_err' : err,
                            'clean_err' : clean_err,
                            'epoch' : t,
                            }, best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))


if __name__ == "__main__":
    args = argparser()
    main(args)
