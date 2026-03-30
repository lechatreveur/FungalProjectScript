#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:32:29 2026

@author: user
"""

import torch

def random_invert(x: torch.Tensor, p: float = 0.5):
    # x: (L,1,H,W) in [0,1]
    if torch.rand(()) < p:
        return 1.0 - x
    return x

def random_rot90(x: torch.Tensor, p: float = 0.3):
    # rotates each tile identically
    if torch.rand(()) < p:
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(x, k=k, dims=(-2, -1))
    return x