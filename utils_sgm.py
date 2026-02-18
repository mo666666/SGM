import numpy as np
import torch
import torch.nn as nn


def backward_hook(gamma):
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_inception(gamma):
    def _backward_hook(module, grad_in, grad_out):
        # if isinstance(module, nn.ReLU):
        return (gamma * grad_in[0],gamma * grad_in[1],gamma * grad_in[2])
    return _backward_hook


def backward_hook_mlp(gamma):
    def _backward_hook(module, grad_in, grad_out):
        return (gamma * grad_in[0], grad_in[1])
    return _backward_hook

def backward_vit_hook(gamma):
    def _backward_hook(module, grad_in, grad_out):
        return (gamma * grad_in[0], grad_in[1])
    return _backward_hook










def register_hook_for_pdart(model, arch, gamma):
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_backward_hook(backward_hook_sgm)


def register_hook_for_inception(model, arch, gamma):
    backward_hook_sgm = backward_hook_inception(gamma)
    for name, module in model.named_modules():
        if 'bn' in name and 'Mixed' in name:
            module.register_backward_hook(backward_hook_sgm)


def register_hook_for_vit(model, arch, gamma):
    if arch in ['vit_base_patch16_224']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_vit_hook(gamma)
    if arch == 'vit_base_patch16_224':
        for blk in model[1].blocks:
            blk.attn.register_backward_hook(backward_hook_sgm)
            blk.mlp.register_backward_hook(backward_hook_sgm)



def register_hook_for_mlp(model, arch, gamma):
    if arch in ['mixer_b16_224']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook_mlp(gamma)
    handle_list = list()
    if arch == 'mixer_b16_224':
        for blk in model[1].blocks:
            blk.mlp_tokens.register_backward_hook(backward_hook_sgm)
            blk.mlp_channels.register_backward_hook(backward_hook_sgm)