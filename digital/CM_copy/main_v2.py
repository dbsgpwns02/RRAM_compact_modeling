import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import random

# 필요한 모듈 import
try:
    from build_feature_v3 import build_features_paper
    from narx_paper import NARX_Paper
except ImportError:
    print("ERROR: build_feature_v3.py 또는 narx_paper.py가 없습니다.")
    sys.exit(1)

# 설정
HERE = Path().resolve()
DATA_DIR = (HERE / "../../data").resolve()
FIGURE_DIR = (HERE / "../FIG").resolve()
TRI_CSV = DATA_DIR / "IV_RRAM_TriD_35_1_real_new2.csv"
CSV_COLS = ["State", "Voltage", "Current"]
ny = 5
Z_DIM = 64
EPOCHS_STEP1 = 10000
EPOCHS_STEP2 = 10000
BATCH_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_iv_csv(path: Path):
    df = pd.read_csv(path, header=None, names=CSV_COLS)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df["State"].values.astype(np.float32), df["Voltage"].values.astype(np.float32), df["Current"].values.astype(np.float32)

def main():
    SEED_VALUE = 42
    set_seed(SEED_VALUE)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"[main] Training Start on {device}")

    # 1. 데이터 로드
    State_tri, V_tri, I_tri = load_iv_csv(TRI_CSV)
    
    # 2. 데이터 증강 (Pristine Ramp-up Oversampling)
    print("[Augment] Injecting 'Pristine' ramp-up data...")
    N_ramp_steps = 10
    V_ramp = np.linspace(0.05, 0.5, N_ramp_steps)
    State_ramp = np.ones_like(V_ramp)
    R_TARGET_HRS = 100e3 # 100k Ohm (정상 HRS)
    I_ramp = V_ramp / R_TARGET_HRS
    
    N_zero_steps = 25
    V_zeros = np.zeros(N_zero_steps); State_zeros = np.ones(N_zero_steps); I_zeros = np.zeros(N_zero_steps)
    
    V_aug = np.concatenate([V_zeros, V_ramp])
    State_aug = np.concatenate([State_zeros, State_ramp])
    I_aug = np.concatenate([I_zeros, I_ramp])

    N_OVERSAMPLE = 500 # 500배 복제
    V_aug_tiled = np.tile(V_aug, N_OVERSAMPLE)
    State_aug_tiled = np.tile(State_aug, N_OVERSAMPLE)
    I_aug_tiled = np.tile(I_aug, N_OVERSAMPLE)
    
    V_tri = np.concatenate([V_aug_tiled, V_tri])
    State_tri = np.concatenate([State_aug_tiled, State_tri])
    I_tri = np.concatenate([I_aug_tiled, I_tri])
    print(f"[Augment] Added {len(V_aug_tiled)} samples.")

    # 3. 특징 생성
    X_tri, Y_R_log_tri, Y_digital_tri, _, _ = build_features_paper(State_tri, V_tri, I_tri, ny=ny)
    
    scX = StandardScaler().fit(X_tri)
    X_tri_scaled = scX.transform(X_tri)

    ds_all = TensorDataset(
        torch.from_numpy(X_tri_scaled).to(device),
        torch.from_numpy(Y_R_log_tri).to(device),
        torch.from_numpy(Y_digital_tri).to(device)
    )
    dl_all = DataLoader(ds_all, batch_size=BATCH_SIZE, shuffle=True)
    
    model = NARX_Paper(in_dim=X_tri.shape[1]).to(device)

    # Step 1: State Module
    print(f"\n[Step 1] Training State Module... ({EPOCHS_STEP1})")
    model.head_hr.requires_grad_(False); model.head_lr.requires_grad_(False)
    model.head_ref.requires_grad_(True); model.z_backbone.requires_grad_(True)
    
    opt1 = optim.Adam(list(model.z_backbone.parameters()) + list(model.head_ref.parameters()), lr=1e-3)
    loss_fn1 = nn.BCEWithLogitsLoss()
    
    for ep in range(1, EPOCHS_STEP1 + 1):
        for xb, _, yb_digital in dl_all:
            r_ref_logit, _, _ = model(xb)
            loss = loss_fn1(r_ref_logit, yb_digital)
            opt1.zero_grad(); loss.backward(); opt1.step()
        if ep % 2000 == 0: print(f" Ep {ep}: Loss {loss.item():.4e}")

    # Step 2: R Modules (z_backbone 미세 조정)
    print(f"\n[Step 2] Training R Modules (Fine-tuning)... ({EPOCHS_STEP2})")
    model.head_ref.requires_grad_(False)
    model.head_hr.requires_grad_(True); model.head_lr.requires_grad_(True)
    model.z_backbone.requires_grad_(True) # [!] 미세 조정 허용

    # 데이터 분리
    hrs_mask = (Y_digital_tri.squeeze() == 1)
    lrs_mask = (Y_digital_tri.squeeze() == 0)
    
    X_hrs = X_tri_scaled[hrs_mask]; Y_R_hrs = Y_R_log_tri[hrs_mask]
    X_lrs = X_tri_scaled[lrs_mask]; Y_R_lrs = Y_R_log_tri[lrs_mask]
    
    dl_hrs = DataLoader(TensorDataset(torch.from_numpy(X_hrs).to(device), torch.from_numpy(Y_R_hrs).to(device)), batch_size=BATCH_SIZE, shuffle=True)
    dl_lrs = DataLoader(TensorDataset(torch.from_numpy(X_lrs).to(device), torch.from_numpy(Y_R_lrs).to(device)), batch_size=BATCH_SIZE, shuffle=True)
    
    opt2 = optim.Adam(
        list(model.z_backbone.parameters()) + list(model.head_hr.parameters()) + list(model.head_lr.parameters()),
        lr=1e-4 # 미세 조정을 위해 학습률 낮춤
    )
    loss_fn2 = nn.MSELoss()

    for ep in range(1, EPOCHS_STEP2 + 1):
        for xb, yb in dl_hrs:
            _, log_hr, _ = model(xb)
            loss = loss_fn2(log_hr, yb)
            opt2.zero_grad(); loss.backward(); opt2.step()
            
        for xb, yb in dl_lrs:
            _, _, log_lr = model(xb)
            loss = loss_fn2(log_lr, yb)
            opt2.zero_grad(); loss.backward(); opt2.step()
            
        if ep % 2000 == 0: print(f" Ep {ep}: Loss Updated")

    # 저장
    model.eval()
    with torch.no_grad():
        X_torch = torch.from_numpy(X_tri_scaled).to(device)
        r_ref = model(X_torch)[0]
        R_ref_minmax = (r_ref.min().item(), r_ref.max().item())

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_scX': scX,
        'R_ref_minmax': R_ref_minmax,
        'ny': ny,
        'in_dim': X_tri.shape[1]
    }, 'checkpoint.pth')
    print("[Save] checkpoint.pth saved.")

if __name__ == "__main__":
    main()