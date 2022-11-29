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



def normalize(X):
    X = (X[..., None].view(X.real.dtype) ** 2).sum(-1)
    POS = int(X.size * 0.99903)
    EXP = norm.ppf((POS + 0.4) / (X.size + 0.215))
    scale = np.partition(X.flatten(), POS, -1)[POS]
    X /= scale / EXP.astype(scale.dtype) ** 2
    return X
