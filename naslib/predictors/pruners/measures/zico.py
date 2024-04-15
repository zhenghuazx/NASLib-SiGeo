import os, sys
import torch
from torch import nn
import numpy as np
from collections import defaultdict

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
def get_eigen(model:torch.nn.Module):
    top_k_eigenvalues = []
    for name,mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            # print(mod.weight.grad.data.size())
            # print(mod.weight.data.size())
            grad = mod.weight.grad.data.cpu().reshape(-1)
            try:
                top_k_eigen, _ = torch.lobpcg(torch.outer(grad, grad), k=2, largest=True)
                top_k_eigen = torch.log(top_k_eigen[0] / top_k_eigen[-1])
                top_k_eigenvalues.append(top_k_eigen.item())
            except:
                continue
    return top_k_eigenvalues

def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


def warmup_net(net, optimizer, inputs, targets, loss_fn, device, step_size=128, epoch=10):
    num_sample = inputs.shape[0]
    for _ in range(epoch):
        for i in range(0, num_sample, step_size):
            # network_weight_gaussian_init(net)
            optimizer.zero_grad()
            data, label = inputs[i:(i+step_size)], targets[i:(i+step_size)]
            data, label = data.to(device), label.to(device)
            # print(f'loss backward start {data}')
            logits = net(data)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
    return net, optimizer

    

@measure("zico", bn=True, )
def get_fr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
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
    
    if num_sample > 256:
        # warm up the net
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        net, optimizer = warmup_net(net, optimizer, inputs, targets, loss_fn, device)
    
    inputs = inputs[-step_size*4:]
    targets = targets[-step_size*4:]
    num_sample = inputs.shape[0]
    for i in range(0, num_sample, step_size):
        net.zero_grad()
        data, label = inputs[i:(i+step_size)], targets[i:(i+step_size)]
        data, label = data.to(device), label.to(device)
        logits = net(data)
        loss = loss_fn(logits, label)
        loss.backward()
        grad_dict = getgrad(net, grad_dict, i)

    score = caculate_zico(grad_dict)

    # score = caculate_zico(grad_dict)
    return score
