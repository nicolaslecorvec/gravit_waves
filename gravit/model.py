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

from gravit.preprocess import LargeKernel_debias, preprocess



"""
This is a test

List of model to test :
    https://smp.readthedocs.io/en/latest/encoders_timm.html

tf_efficientnet_l2_ns	88.35%	98.66%
tf_efficientnet_b7_ns	86.83%	98.08%
tf_efficientnet_b6_ns	86.45%	97.88%
tf_efficientnet_b5_ns	86.08%	97.75%
tf_efficientnet_b4_ns	85.15%	97.47%
tf_efficientnet_b3_ns	84.04%	96.91%
tf_efficientnet_b2_ns	82.39%	96.24%
tf_efficientnet_b1_ns	81.39%	95.74%
tf_efficientnet_b0_ns

List to tester :
    Adam
    softmax
"""

def get_model(path):
    model = create_model(
        "tf_efficientnetv2_b0",
        in_chans=32,
        num_classes=2,
    )
    state_dict = torch.load(path)
    C, _, H, W = state_dict["conv_stem.2.weight"].shape
    model.conv_stem = nn.Sequential(
        nn.Identity(),
        nn.AvgPool2d((1, 9), (1, 8), (0, 4), count_include_pad=False),
        LargeKernel_debias(1, C, [H, W], 1, [H//2, W//2], 1, 1, False),
        model.conv_stem,
    )
    model.load_state_dict(state_dict)
    model.cuda().eval()
    return model
