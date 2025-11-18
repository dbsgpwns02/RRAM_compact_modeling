# evaluate.py (신규 파일)

import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler # 로드에 필요

# [필수] 1. 필요한 함수들 import
from generate_pwl import generate_v_data_from_spice_pwl
from export_narx_to_veriloga_paper import export_veriloga_hspice_split
from narx_paper_v2 import NARX_Paper # 모델 껍데기 import
from plot_transient import plot_transient
from debug_sample_like_v3 import debug_spice_sampling_va_matched_CONTINUOUS

# ------------------------------------------------------------
# 경로 & 설정 (기존 main.py와 동일하게)
# ------------------------------------------------------------
HERE = Path().resolve()
FIGURE_DIR = (HERE / "../FIG").resolve()
EXPORT_DIR = (HERE / "../VA").resolve()
CHECKPOINT_PATH = 'checkpoint.pth' # 불러올 파일

# --- 모델 및 시뮬레이션 파라미터 (기존과 동일) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 40
R_clip_min = 100.0

# ------------------------------------------------------------

def main_evaluate():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"[main] Device set to: {device}")

    # =========================================================
    # 1. 체크포인트 로드 (훈련 생략)
    # =========================================================
    print(f"\n[Load] Loading checkpoint from '{CHECKPOINT_PATH}'...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[Error] '{CHECKPOINT_PATH}' not found. Please run train.py first.")
        sys.exit(1)
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print("[Load] Checkpoint loaded.")

    # =========================================================
    # 2. 모델 및 스케일러 복원
    # =========================================================
    # 저장된 정보로 모델 껍데기 생성
    ny = checkpoint['ny']
    in_dim = checkpoint['in_dim']
    model = NARX_Paper(in_dim=in_dim).to(device)
    
    # 가중치, 스케일러, 파라미터 복원
    model.load_state_dict(checkpoint['model_state_dict'])
    scX = checkpoint['scaler_scX']
    R_ref_minmax = checkpoint['R_ref_minmax']
    
    model.eval() # [중요] 추론 모드로 설정
    print("[Load] Model and scaler restored successfully.")

    # =========================================================
    # 3. Verilog-A 파일로 익스포트 (기존 3번 블록)
    # =========================================================
    print("\n[Export] Exporting trained model to Verilog-A files...")
    export_veriloga_hspice_split(
        model=model,
        scX=scX,
        R_ref_minmax=R_ref_minmax,
        ny=ny,
        out_base=str(EXPORT_DIR / "narx_rram"),
        module_name="narx_rram",
        Ts=1e-9,
        time_step=1e-9,
        init_w=1,
        warmup_steps=WARMUP_STEPS,
        r_clip_min=100.0,
        r_clip_max=1.0e8,
        v_filter=0.01
    )
    print(f"[Export] Files saved to {EXPORT_DIR}/narx_rram_assign.vams and ..._unrolled_math.inc")

    # =========================================================
    # 4. SPICE 1:1 PWL 검증 (기존 4번 블록)
    # =========================================================
    print("\n[DEBUG-VA] SPICE 1:1 PWL 검증 시작...")

    # --- [제어판] ---
    TARGET_TAU_SLEW = 1e-9 # LPF 끄기 (기존 0으로 설정)
    # -----------------

    # 4-1. SPICE PWL 데이터 생성
    time_data_for_v, V_data_to_test = generate_v_data_from_spice_pwl(
        Ts=1e-9, 
        t_stop=1728e-9,
        save_csv_path=str(FIGURE_DIR / "IV_SPICE_PWL_Input.csv")
    )
    
    # 4-2. 더미 데이터 생성
    I_data_dummy = np.zeros_like(V_data_to_test)
    State_data_dummy = np.zeros_like(V_data_to_test)
    
    # 4-3. VA-Matched 디버깅 함수 호출 (기존 4번 블록과 동일)
    print(f"[DEBUG-VA] Static SPICE PWL, tau={TARGET_TAU_SLEW:.1e} s 시뮬레이션 실행...")
    
    time_result, V_result, I_result, R_result, R_REF_result, I_raw_result, alpha_result, log_hr_result, log_lr_result= debug_spice_sampling_va_matched_CONTINUOUS (
        model=model,
        V_data=V_data_to_test,
        I_data=I_data_dummy,
        State_data=State_data_dummy,
        scX=scX,
        ny=ny,
        R_ref_minmax=R_ref_minmax,
        device=device,
        Ts=1e-9,
        V_FILTER=0.01, # V_READ_TH 로직으로 변경되어 실제 사용 안함
        warmup_steps=WARMUP_STEPS,
        INIT_W=1,
        tau_slew=TARGET_TAU_SLEW,
        r_clip_min=R_clip_min,
        save_log=True,
        log_path=str(FIGURE_DIR / "FINAL_SPICE_vs_PYTHON.txt"),
        V_smooth_read=0.1,
        hyst_smooth=0.1
    )
    
    print(f"\n[DEBUG-VA] 검증 완료.")
    print(f"로그 파일: {FIGURE_DIR / 'FINAL_SPICE_vs_PYTHON.txt'}")

    # =========================================================
    # 5. 결과 그래프 (기존 5번 블록)
    # =========================================================
    print("\n[Plotting] Saving transient results...")
    
    # Time-V (Stimulus) 그래프
    plot_transient(time_data_for_v, V_data_to_test, 
                   f"Stimulus Voltage (Static SPICE PWL)", 
                   "Voltage (V)", 
                   FIGURE_DIR / "transient_voltage_predicted.png")
    
    # Time-I (Predicted Current) 그래프
    # [수정] tau=0.0일 때는 I_raw_result (사각)를 그리는 것이 좋음
    plot_transient(time_result, I_raw_result, 
                   f"Current Transient (Raw, tau={TARGET_TAU_SLEW:.1e} s)", 
                   "Current (A)", 
                   FIGURE_DIR / "transient_current_predicted.png")
    
    # Time-r_ref_logit (Digital State) 그래프
    plot_transient(time_result, R_REF_result, 
                   "Internal State (r_ref_logit) Transient", 
                   "r_ref_logit (State)", 
                   FIGURE_DIR / "transient_r_ref_logit_predicted.png",
                   add_zero_line=True)

    print(f"\n[Done] All outputs saved to {FIGURE_DIR}")

if __name__ == "__main__":
    main_evaluate()