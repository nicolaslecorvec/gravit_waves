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


from gravit.normalize import normalize

normalize = normalize()

##importer la data
def dataload(filepath):
    astime = np.full([2, 360, 5760], np.nan, dtype=np.float32)
    with h5py.File(filepath, "r") as f:
        fid, _ = os.path.splitext(os.path.split(filepath)[1])
        HT = (np.asarray(f[fid]["H1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        LT = (np.asarray(f[fid]["L1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        MIN = min(HT.min(), LT.min()); HT -= MIN; LT -= MIN
        H1 = normalize(np.asarray(f[fid]["H1"]["SFTs"], np.complex128))
        valid = HT < 5760; astime[0][:, HT[valid]] = H1[:, valid]
        L1 = normalize(np.asarray(f[fid]["L1"]["SFTs"], np.complex128))
        valid = LT < 5760; astime[1][:, LT[valid]] = L1[:, valid]
    gc.collect()
    return fid, astime, H1.mean(), L1.mean()


@torch.no_grad()
def inference(model, path):
    file_path = glob.glob(os.path.join(path, "*.hdf5"))
    FID, RES = [], []
    with ProcessPoolExecutor(2) as pool:
        for fid, input, H1, L1 in pool.map(dataload, sorted(file_path)):
            tta = preprocess(64, input, H1, L1)
            FID += [fid]
            RES += [model(tta).softmax(-1)[..., 1].mean(0)]
    return FID, torch.stack(RES, 0).cpu().float().numpy()
