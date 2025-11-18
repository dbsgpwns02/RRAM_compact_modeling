import numpy as np
import torch

def debug_spice_sampling_va_matched_CONTINUOUS(
    model, V_data, I_data, State_data, scX, ny, R_ref_minmax,
    device, 
    Ts=1e-9, V_FILTER=0.01, 
    warmup_steps=40, INIT_W=1,
    tau_slew=0.0, r_clip_min=100.0,
    save_log=True, log_path="spice_debug.txt",
    V_smooth_read=0.02, hyst_smooth=0.1
):
    V_READ_TH = 0.6
    model.eval()

    t_stop = (len(V_data) - 1) * Ts
    time_array = np.arange(0, t_stop + Ts/2, Ts)[:len(V_data)]

    V_prev_raw = float(V_data[0])
    R_last = float(r_clip_min)
    G_last = 1.0 / R_last
    k_eff = 0
    W_hist = [float(INIT_W)] * ny
    
    R_ref_min, R_ref_max = R_ref_minmax
    if R_ref_max == R_ref_min: R_ref_max += 1e-9
    r_ref_last_valid = R_ref_max if INIT_W == 1 else R_ref_min
    
    I_hold = 0.0
    alpha_slew = 1.0 if tau_slew <= 0.0 else min(1.0, Ts/(tau_slew+Ts))
    
    all_I, all_I_raw = [], []
    # (나머지 리스트 생략 가능하지만 구조 유지를 위해)
    all_t, all_V, all_R, all_r_ref, all_alpha, all_log_hr, all_log_lr = [],[],[],[],[],[],[]

    for t_idx, t_now in enumerate(time_array):
        Vt = float(V_data[t_idx])
        x_raw = np.array([Vt, np.sign(Vt-V_prev_raw), abs(Vt-V_prev_raw)] + W_hist, dtype=np.float32).reshape(1,-1)
        x_t = torch.from_numpy(scX.transform(x_raw)).to(device).float()
        
        with torch.no_grad():
            (I_t, R_t, _, r_logit, log_hr, log_lr, alpha) = model.predict_with_internals(x_t, torch.tensor(Vt).to(device), R_ref_minmax)
        
        # Hybrid Logic
        if abs(Vt) < V_FILTER:
            I_raw = Vt * G_last
            R_curr = R_last
            r_logit = r_ref_last_valid
        else:
            gate_W = 0.5 * (1.0 + np.tanh((abs(Vt) - V_READ_TH) / V_smooth_read))
            
            I_read = Vt * G_last
            R_read = R_last
            r_read = r_ref_last_valid
            
            I_raw = (1-gate_W)*I_read + gate_W*I_t.item()
            R_curr = (1-gate_W)*R_read + gate_W*R_t.item()
            r_logit = (1-gate_W)*r_read + gate_W*r_logit.item()
            
            W_targ = 0.5 * (1.0 + np.tanh(r_logit / hyst_smooth))
            W_next = (1-gate_W)*W_hist[0] + gate_W*W_targ
            
            if k_eff >= warmup_steps:
                W_hist = [W_next] + W_hist[:-1]
            k_eff += 1

        # LPF
        I_hold += alpha_slew * (I_raw - I_hold)
        
        # [!] [강제 0 로직] 루프 내부에서 변수 덮어쓰기
        if t_now < 60e-9:
            I_hold = 0.0
            I_raw = 0.0
            R_curr = R_ref_max if INIT_W==1 else R_ref_min
            W_hist = [float(INIT_W)] * ny # 히스토리도 초기화 유지
            
        R_last = R_curr
        G_last = 1.0/R_curr if R_curr!=0 else 0.0
        r_ref_last_valid = r_logit
        V_prev_raw = Vt

        all_t.append(t_now); all_V.append(Vt); all_I.append(I_hold); all_I_raw.append(I_raw)
        all_R.append(R_curr); all_r_ref.append(r_logit); all_alpha.append(alpha.item()); 
        all_log_hr.append(log_hr.item()); all_log_lr.append(log_lr.item())

    return (np.array(all_t), np.array(all_V), np.array(all_I), np.array(all_R), np.array(all_r_ref),
            np.array(all_I_raw), np.array(all_alpha), np.array(all_log_hr), np.array(all_log_lr))