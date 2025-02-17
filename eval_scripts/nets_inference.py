"""
This file defines a single torch-module for computing the forward-pass of
several networks on the same batch of inputs.
"""

import torch
from .vit_models import deit_base_patch16_LS
from torchvision import models
import os
from torch import nn


class Nets(torch.nn.Module):

    def __init__(self, paths, indices, RANK=0):
        super().__init__()

        def make_net(key):
            if key[:3] == 'VIT':
                net = deit_base_patch16_LS().cuda(RANK).eval()
                if os.path.exists(paths[key]):
                    ckpt = torch.load(paths[key])
                    net.load_state_dict(ckpt['model'])
                    print("Loaded DeIT+DiffAug")
                else:
                    ckpt = torch.load("workdirs/deit_3_base_224_1k.pth")
                    net.load_state_dict(ckpt['model'])
                    print("Loaded default DeIT")
            else:
                net = torch.nn.DataParallel(
                    models.resnet50(weights='IMAGENET1K_V1').cuda(RANK),
                    device_ids=[RANK]).eval()
                if os.path.exists(paths[key]):
                    ckpt = torch.load(paths[key])
                    if 'state_dict' in ckpt:
                        net.load_state_dict(ckpt['state_dict'])
                    print(f"Loaded {paths[key]} model")
                else:
                    print("Loaded default RN50 model")
            return net

        self.nets = nn.ModuleDict()
        for key in paths:
            self.nets[key] = make_net(key)
        self.indices = indices

    def forward(self, images):
        out = dict()
        for key in self.nets:
            if key[:3] != 'VIT':
                out[key] = self.nets[key](images).cpu()
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    out[key] = self.nets[key](images)
                out[key] = out[key].float().cpu()
            if self.indices is not None:
                out[key] = out[key][:, self.indices]
        return out


def get_nets(indices=None):
    import os
    print(os.listdir())
    orig_net = Nets(
        {
            "DA": "workdirs/deepaugment.pth.tar",
            "AM": "workdirs/checkpoint.pth.tar",
            "DAM": "workdirs/deepaugment_and_augmix.pth.tar",
            "BASE": "NONE",
            "VIT": "NONE",
        }, indices)
    
    diffaug_net = Nets(
        {
            "DA": "workdirs/DA_diffaug.ckpt",
            "AM": "workdirs/AM_diffaug.ckpt",
            "DAM": "workdirs/DAM_diffaug.ckpt",
            "BASE": "workdirs/BASE_diffaug.ckpt",
            "VIT": "workdirs/VIT_diffaug.ckpt",
        }, indices)

    return orig_net, diffaug_net
