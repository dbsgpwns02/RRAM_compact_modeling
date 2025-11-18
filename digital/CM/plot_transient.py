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

def plot_transient(time_steps, data_values, title, ylabel, save_path, add_zero_line=False):
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps * 1e9, data_values, c='blue', label='Predicted')
    
    if add_zero_line:
        plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (y=0)')
        
    plt.title(title)
    plt.xlabel("Time (ns)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")