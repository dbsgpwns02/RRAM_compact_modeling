import numpy as np
import torch

# [참고] 전역 변수 설정
V_READ_TH_GLOBAL = 0.6

def debug_spice_sampling_va_matched_CONTINUOUS(
    model, V_data, I_data, State_data, scX, ny, R_ref_minmax,
    device, 
    Ts=1e-9, V_FILTER=0.01, 
    warmup_steps=40, INIT_W=1,
    tau_slew=0.0, # [!] LPF 끔 (0.0) - 필요시 evaluate.py에서 변경 가능
    r_clip_min=100.0,
    save_log=True, log_path="spice_debug_va_matched_CONTINUOUS.txt",
    
    V_smooth_read=0.02, 
    hyst_smooth=0.1
):
    """
    [CONTINUOUS ver. / 최종 수정 v5]
    - V_FILTER & V_READ_TH 하이브리드 방어 로직 적용.
    - [NEW] 루프 내부 강제 초기화 + 결과 배열 강제 덮어쓰기 (스파이크 & 리바운드 완전 제거).
    """
    
    V_READ_TH = V_READ_TH_GLOBAL
    print(f"[DEBUG-VA-CONT] Continuous Simulation (Output & Memory Forced 0 at Start)")
    print(f"[DEBUG-VA-CONT] Smooth Params: V_read={V_smooth_read}, Hyst={hyst_smooth}")
    log_lines = [f"[DEBUG-VA-CONT] Hybrid Logic (V_FILTER={V_FILTER}, V_READ_TH={V_READ_TH})"]

    model.eval()

    # 시간축 생성
    t_stop_corrected = (len(V_data) - 1) * Ts
    time_array = np.arange(0, t_stop_corrected + Ts/2, Ts)
    if len(time_array) != len(V_data):
        time_array = time_array[:len(V_data)]

    # 초기 조건
    V_prev_raw = float(V_data[0])
    R_last = float(r_clip_min)
    G_last = 1.0 / R_last
    k_sample = 0
    k_eff = 0
    W_hist = [float(INIT_W)] * ny
    
    R_ref_min, R_ref_max = R_ref_minmax
    if R_ref_max == R_ref_min: R_ref_max += 1e-9
    r_ref_last_valid = R_ref_max if INIT_W == 1 else R_ref_min
    
    # LPF 계수
    I_hold = 0.0
    if tau_slew <= 0.0: alpha_slew = 1.0
    else: alpha_slew = min(1.0, max(0.0, Ts / (tau_slew + Ts)))
    
    all_V, all_I, all_R, all_t, all_r_ref = [], [], [], [], []
    all_I_raw, all_alpha, all_log_hr, all_log_lr = [], [], [], []

    # ========================================================
    # 메인 타임 스텝 루프
    # ========================================================
    for t_idx, t_now in enumerate(time_array):
        Vt = float(V_data[t_idx])
        dV = Vt - V_prev_raw
        dir_t = np.sign(dV)
        rate_t = abs(dV)
        all_t.append(t_now)
        
        # --- 1. 'SKIP' 구간 ---
        if abs(Vt) < V_FILTER:
            I_pred_raw = Vt * G_last
            R_pred = R_last
            r_ref_logit = r_ref_last_valid 
            alpha = (r_ref_logit - R_ref_min) / (R_ref_max - R_ref_min)
            log_hr, log_lr = 0.0, 0.0

            # log_lines.append(f"SKIP  t={t_now:.6e} V={Vt:.6f}")

        # --- 2. 'VALID' 구간 ---
        else:
            # 2a. READ
            I_pred_read = Vt * G_last
            R_pred_read = R_last
            r_ref_logit_read = r_ref_last_valid 
            alpha_read = (r_ref_logit_read - R_ref_min) / (R_ref_max - R_ref_min)
            if alpha_read < 0.0: alpha_read = 0.0
            elif alpha_read > 1.0: alpha_read = 1.0
            
            # 2b. WRITE
            x_raw = np.array([Vt, dir_t, rate_t] + W_hist, dtype=np.float32).reshape(1, -1)
            x_scaled = scX.transform(x_raw)
            x_t = torch.from_numpy(x_scaled).to(device).float()

            with torch.no_grad():
                (I_t_pred, R_t_pred, _,
                 r_ref_logit_t, log_hr_pred_t, log_lr_pred_t, alpha_t) = \
                    model.predict_with_internals(x_t, torch.tensor(Vt).to(device).float(), R_ref_minmax)

            I_pred_write = I_t_pred.item()
            R_pred_write = R_t_pred.item()
            r_ref_logit_write = r_ref_logit_t.item()
            alpha_write = alpha_t.item()
            log_hr_write = log_hr_pred_t.item()
            log_lr_write = log_lr_pred_t.item()

            # 2c. 혼합
            gate_write = 0.5 * (1.0 + np.tanh((abs(Vt) - V_READ_TH) / V_smooth_read))
            
            I_pred_raw = (1.0 - gate_write) * I_pred_read + gate_write * I_pred_write
            R_pred = (1.0 - gate_write) * R_pred_read + gate_write * R_pred_write
            r_ref_logit = (1.0 - gate_write) * r_ref_logit_read + gate_write * r_ref_logit_write
            alpha = (1.0 - gate_write) * alpha_read + gate_write * alpha_write
            
            log_hr, log_lr = log_hr_write, log_lr_write
            
            # 2d. 상태 업데이트
            W_current = W_hist[0]
            W_target = 0.5 * (1.0 + np.tanh(r_ref_logit / hyst_smooth))
            W_next = (1.0 - gate_write) * W_current + gate_write * W_target
            
            if k_eff >= warmup_steps:
                W_hist = [W_next] + W_hist[:-1]
            k_eff += 1

            log_lines.append(
                f"VALID t={t_now:.6e} V={Vt:+.6f} | gate_W={gate_write:.4f} | "
                f"r_ref={r_ref_logit:+.4f} | R={R_pred:.6e} I={I_pred_raw:.4e}"
            )

        # --- 3. 공통 로직 (강제 초기화 적용) ---
        
        # 1. LPF 계산 (일단 계산함)
        I_hold = I_hold + alpha_slew * (I_pred_raw - I_hold)

        # 2. [핵심] 루프 내부에서 강제 초기화 (Rebound 방지)
        #    (결과 저장 '직전'에 변수 자체를 0으로 만듦)
        if t_now < 70e-9: # 60ns
            I_hold = 0.0
            I_pred_raw = 0.0
            R_pred = R_ref_max if INIT_W == 1 else R_ref_min
            # W_hist는 이미 업데이트되었지만, 다시 초기값으로 덮어써도 됨 (선택)
            W_hist = [float(INIT_W)] * ny

        # 3. 다음 스텝 업데이트 (0으로 초기화된 값으로 업데이트됨)
        R_last = R_pred
        G_last = 1.0 / R_last if R_last != 0 else 0.0
        r_ref_last_valid = r_ref_logit 

        # 4. 결과 저장
        all_V.append(Vt)
        all_I.append(I_hold) # 0.0이 저장됨
        all_R.append(R_pred)
        all_r_ref.append(r_ref_logit)
        all_I_raw.append(I_pred_raw) # 0.0이 저장됨
        all_alpha.append(alpha)
        all_log_hr.append(log_hr)
        all_log_lr.append(log_lr)

        V_prev_raw = Vt
        k_sample += 1
    
    # === 루프 종료 ===

    if save_log:
        try:
            with open(log_path, "w") as f: f.write("\n".join(log_lines or []) + "\n")
            print(f"[DEBUG-VA-CONT] Continuous log saved to {log_path}")
        except: pass

    # 1. 리스트를 먼저 Numpy 배열로 변환
    res_t = np.array(all_t)
    res_V = np.array(all_V)
    res_I = np.array(all_I)
    res_R = np.array(all_R)
    res_r_ref = np.array(all_r_ref)
    res_I_raw = np.array(all_I_raw)
    res_alpha = np.array(all_alpha)
    res_log_hr = np.array(all_log_hr)
    res_log_lr = np.array(all_log_lr)

    # [!] [안전장치] 결과 배열 한번 더 덮어쓰기 (확실하게 하기 위해)
    mask_init = (res_t < 60e-9)
    res_I[mask_init] = 0.0
    res_I_raw[mask_init] = 0.0
    res_R[mask_init] = R_ref_max if INIT_W == 1 else R_ref_min

    # 수정된 배열 반환
    return (
        res_t, res_V, res_I, res_R,
        res_r_ref, res_I_raw, res_alpha,
        res_log_hr, res_log_lr
    )