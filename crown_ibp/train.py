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
from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
from convex_adversarial import DualNetwork
from eps_scheduler import EpsilonScheduler
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


def Train(model, t, loader, eps_scheduler, max_eps, norm, logger, verbose, train, opt, method, **kwargs):
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
    batch_multiplier = kwargs.get("batch_multiplier", 1)  
    kappa = 1
    beta = 1
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
    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
 
    model_range = 0.0
    end_eps = eps_scheduler.get_eps(t+1, 0)
    if end_eps < np.finfo(np.float32).tiny:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
    for i, (data, labels) in enumerate(loader): 
        start = time.time()
        eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
        if train and i % batch_multiplier == 0:   
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
        ub_s = torch.zeros(data.size(0), num_class)

        # FIXME: Assume unnormalized data is from range 0 - 1
        if kwargs["bounded_input"]:
            if norm != np.inf:
                raise ValueError("bounded input only makes sense for Linf perturbation. "
                                 "Please set the bounded_input option to false.")
            data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / std), data_max)
            data_lb = torch.max(data - (eps / std), data_min)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                # For other norms, eps will be used instead.
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
            ub_s = ub_s.cuda()
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps

        # omit the regular cross entropy, since we use robust error
        output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural"))
        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), data.size(0))
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()
        
        '''
        torch.set_printoptions(threshold=5000)
        print('prediction:  ', output)
        ub, lb, _, _, _, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('interval ub: ', ub)
        print('interval lb: ', lb)
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('crown-ibp ub: ', ub)
        print('crown-ibp lb: ', lb) 
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="full_backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('full-crown ub: ', ub)
        print('full-crown lb: ', lb)
        input()
        '''
        

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
                        assert loader.std == [1,1,1] or loader.std == [1]
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
                ub, lb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
            elif kwargs["bound_type"] == "crown-full":
                _, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=False, lower=True, method_opt="full_backward_range")
                unstable = dead = alive = relu_activity = torch.tensor([0])
            elif kwargs["bound_type"] == "crown-interval":
                # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
                # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 
                ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
                crown_final_beta = kwargs['final-beta']
                beta = (max_eps - eps * (1.0 - crown_final_beta)) / max_eps
                if beta < 1e-5:
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
                        _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                        clb = clb.expand(clb.size(0), num_class - 1)
                    else:
                        # get the CROWN bound using interval bounds 
                        _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                        bound_bias.update(bias.sum() / data.size(0))
                    # how much better is crown-ibp better than ibp?
                    diff = (clb - ilb).sum().item()
                    bound_diff.update(diff / data.size(0), data.size(0))
                    # lb = torch.max(lb, clb)
                    lb = clb * beta + ilb * (1 - beta)
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"]) 
            lb = lb_s.scatter(1, sa_labels, lb)
            robust_ce = CrossEntropyLoss()(-lb, labels)
            if kwargs["bound_type"] != "convex-adv":
                
                relu_activities.update(relu_activity.sum().detach().cpu().item() / data.size(0), data.size(0))
                unstable_neurons.update(unstable.sum().detach().cpu().item() / data.size(0), data.size(0))
                dead_neurons.update(dead.sum().detach().cpu().item() / data.size(0), data.size(0))
                alive_neurons.update(alive.sum().detach().cpu().item() / data.size(0), data.size(0))

        if method == "robust":
            loss = robust_ce
        elif method == "robust_activity":
            loss = robust_ce + kwargs["activity_reg"] * relu_activity.sum()
        elif method == "natural":
            loss = regular_ce
        elif method == "robust_natural":
            natural_final_factor = kwargs["final-kappa"]
            kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            loss = (1-kappa) * robust_ce + kappa * regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if train and kwargs["l1_reg"] > np.finfo(np.float32).tiny:
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + torch.sum(torch.abs(param))
            l1_loss = reg * l1_loss
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))
        if train:
            loss.backward()
            if i % batch_multiplier == 0 or i == len(loader) - 1:
                opt.step()

        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            # robust_ce_losses.update(robust_ce, data.size(0))
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))

        batch_time.update(time.time() - start)
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
                    'beta {beta:.3f} ({beta:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
                    unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
                    tight = relu_activities, bias = bound_bias, diff = bound_diff,
                    model_range = model_range, 
                    beta=beta, kappa = kappa))
    
                    
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
        'beta {beta:.3f} ({beta:.3f})  '
        'kappa {kappa:.3f} ({kappa:.3f})  \n'.format(
        t, eps, batch_time=batch_time,
        loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
        regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
        unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
        tight = relu_activities, bias = bound_bias, diff = bound_diff,
        model_range = model_range, 
        kappa = kappa, beta=beta))
    for i, l in enumerate(model if isinstance(model, BoundSequential) else model.module):
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
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])
        model = BoundSequential.convert(model, train_config["method_params"]["bound_opts"])
        
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
        lr_decay_milestones = train_config["lr_decay_milestones"]
        lr_decay_factor = train_config["lr_decay_factor"]
        multi_gpu = train_config["multi_gpu"]
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
       
        batch_multiplier = train_config["method_params"].get("batch_multiplier", 1)
        batch_size = train_data.batch_size * batch_multiplier  
        num_steps_per_epoch = int(np.ceil(1.0 * len(train_data.dataset) / batch_size))
        epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"), schedule_start * num_steps_per_epoch, ((schedule_start + schedule_length) - 1) * num_steps_per_epoch, starting_epsilon, end_epsilon, num_steps_per_epoch)
        max_eps = end_epsilon
        
        if lr_decay_step:
            # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
            lr_decay_milestones = None
        elif lr_decay_milestones:
            # Decay learning rate by lr_decay_factor at a few milestones.
            lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones, gamma=lr_decay_factor)
        else:
            raise ValueError("one of lr_decay_step and lr_decay_milestones must be not empty.")
        model_name = get_path(config, model_id, "model", load = False)
        best_model_name = get_path(config, model_id, "best_model", load = False) 
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log(model_name)
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0
         
        if multi_gpu:
            logger.log("\nUsing multiple GPUs for computing CROWN-IBP bounds\n")
            model = BoundDataParallel(model) 
        model = model.cuda()
        
        for t in range(epochs):
            epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
            epoch_end_eps = epsilon_scheduler.get_eps(t+1, 0)
            logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(), epoch_start_eps, epoch_end_eps))
            # with torch.autograd.detect_anomaly():
            start_time = time.time() 
            Train(model, t, train_data, epsilon_scheduler, max_eps, norm, logger, verbose, True, opt, method, **method_param)
            if lr_decay_step:
                # Use stepLR. Note that we manually set up epoch number here, so the +1 offset.
                lr_scheduler.step(epoch=max(t - (schedule_start + schedule_length - 1) + 1, 0))
            elif lr_decay_milestones:
                # Use MultiStepLR with milestones.
                lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            with torch.no_grad():
                # evaluate
                err, clean_err = Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1), max_eps, norm, logger, verbose, False, None, method, **method_param)

            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                    'epoch' : t,
                    }, model_name)

            # save the best model after we reached the schedule
            if t >= (schedule_start + schedule_length):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                            'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                            'robust_err' : err,
                            'clean_err' : clean_err,
                            'epoch' : t,
                            }, best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))


if __name__ == "__main__":
    args = argparser()
    main(args)
