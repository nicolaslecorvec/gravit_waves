import numpy as np
import pandas as pd
import numpy.fft as fft
import matplotlib.pyplot as plt
import pickle
import os
import math
import h5py
import random
import cv2
import gc
from tqdm import tqdm
from multiprocessing import Pool
import gc, glob, os
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
from scipy.stats import norm
from timm import create_model





class LargeKernel_debias(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)
        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2 + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)
        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        output.sub_(power.mul_(ratio))
        return output.unflatten(0, input.shape[:2]).flatten(1, 2)



def preprocess(num, input, H1, L1):
    input = torch.from_numpy(input).to("cuda", non_blocking=True)
    rescale = torch.tensor([[H1, L1]]).to("cuda", non_blocking=True)
    tta = (
        torch.randn(
            [num, *input.shape, 2], device=input.device, dtype=torch.float32
        )
        .square_()
        .sum(-1)
    )
    tta *= rescale[..., None, None] / 2
    valid = ~torch.isnan(input); tta[:, valid] = input[valid].float()
    return tta
