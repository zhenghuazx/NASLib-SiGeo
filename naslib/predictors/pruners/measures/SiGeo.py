import os, sys
import torch
from torch import nn
import numpy as np
from collections import defaultdict
import copy
from sklearn.linear_model import LinearRegression
# from functorch import make_functional, jacfwd, jacrev, vmap

from . import measure


def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())

    return grad_dict


def caculate_sigeo(grad_dict, theta_dict, losses, theta_dict_copy=None):
    ''' Use implementation based on zico because the module names of search spaces in the
        CV benchmark are different from the ads search space.
    '''
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    per_sample_mean_abs = np.zeros(4)
    nsr_mean_sum_abs = 0
    nsr_mean_sum_mean = 0
    nsr_mean_sum_std = 0
    fr_mean_sum_abs = 0
    per_sample_prod_grad = np.zeros(4)

    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        temp = np.mean(np.abs(grad_dict[modname])[:, nonzero_idx], axis=1)

        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_sum_grad = np.sum(np.sum(grad_dict[modname], axis=0) * theta_dict[-1][modname])
            fr_mean_sum_abs += np.abs(nsr_sum_grad)

    return nsr_mean_sum_abs, np.log(fr_mean_sum_abs + 1e-5) * 2


def caculate_var(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        tmpsum = np.sum(1/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
    return nsr_mean_sum_abs


def warmup_net(net, optimizer, inputs, targets, loss_fn, device, step_size=128, epoch=10):
    num_sample = inputs.shape[0]
    for _ in range(epoch):
        for i in range(0, num_sample, step_size):
            optimizer.zero_grad()
            data, label = inputs[i:(i+step_size)], targets[i:(i+step_size)]
            data, label = data.to(device), label.to(device)
            # print(f'loss backward start {data}')
            logits = net(data)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
    return net, optimizer


@measure("SiGeo", bn=True, )
def get_sigeo(net, inputs, targets, loss_fn, split_data=1, skip_grad=False, lambda2=1, lambda3=0):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
         device = torch.device("cpu")
    num_sample = inputs.shape[0]
    step_size = 64
    grad_dict= {}
    net.to(device)
    net.train()

    num_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            num_param += param.flatten().size()[0]
    scores = torch.zeros(num_param).to('cpu')
    optimizer = torch.optim.SGD(net.parameters(), lr=1)
    
    # warm up the net
    if num_sample > 256:
        lambda2 = 50
        lambda3 = 1
        net, optimizer = warmup_net(net, optimizer, inputs, targets, loss_fn, device)
    
    inputs = inputs[-step_size*4:]
    targets = targets[-step_size*4:]
    num_sample = inputs.shape[0]
    losses = []
    theta_list = []
    for i in range(0, num_sample, step_size):
        optimizer.zero_grad()
        data, label = inputs[i:(i+step_size)], targets[i:(i+step_size)]
        data, label = data.to(device), label.to(device)
        logits = net(data)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        grad_dict= getgrad(net, grad_dict, i)
        losses.append(loss.item())

        theta_dict = {}
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                theta_dict[name] = mod.weight.data.cpu().reshape(-1).numpy()
        theta_list.append(theta_dict)

    score1, score2 = caculate_sigeo(grad_dict, theta_list, np.array(losses))
    print('zico: ', score1, 'fr: ', score2, 'train loss: ', losses[-1])
    return score1 + lambda2 * score2 - lambda3 * losses[-1]
