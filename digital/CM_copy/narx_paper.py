import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import random # 데이터 증강용
ny = 5
Z_DIM = 64
EPOCHS_STEP1 = 10000
EPOCHS_STEP2 = 10000
BATCH_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 40
R_clip_min=100.0 # <<< build_features_paper에서 사용하기 위해 전역으로 이동
class NARX_Paper(nn.Module):
    def __init__(self, in_dim, z_dim=Z_DIM):
        super().__init__()
        self.z_backbone = nn.Sequential(
            nn.Linear(in_dim, z_dim),
            nn.Tanh()
        )
        self.head_ref = nn.Linear(z_dim, 1)
        hr_lr_in_dim = z_dim + 1
        self.head_hr = nn.Sequential(
            nn.Linear(hr_lr_in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.head_lr = nn.Sequential(
            nn.Linear(hr_lr_in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        z = self.z_backbone(x)
        r_ref_logit = self.head_ref(z)
        v_scaled = x[:, 0:1]
        z_and_v = torch.cat([z, v_scaled], dim=-1)
        log_hr = self.head_hr(z_and_v)
        log_lr = self.head_lr(z_and_v)
        return r_ref_logit, log_hr, log_lr
    
    def predict_with_internals(self, x, V_t, R_ref_minmax):
        R_ref_min, R_ref_max = R_ref_minmax
        if R_ref_max == R_ref_min:
            R_ref_max += 1e-9
        r_ref_logit, log_hr_pred, log_lr_pred = self.forward(x)
        hr_val = torch.exp(log_hr_pred)
        lr_val = torch.exp(log_lr_pred)
        alpha = (r_ref_logit - R_ref_min) / (R_ref_max - R_ref_min)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        R_t_pred = alpha * hr_val + (1.0 - alpha) * lr_val
        I_t_pred = V_t / R_t_pred
        next_digital_state = (r_ref_logit > 0).float()
        return (I_t_pred, R_t_pred, next_digital_state,
                r_ref_logit, log_hr_pred, log_lr_pred, alpha)