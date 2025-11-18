#
# export_narx_to_veriloga_paper.py (전체 코드)
# [최종 수정] 1. _assign.vams 파일에서 'assign' 키워드 제거
# [최종 수정] 2. 셸 파일 .vams의 for 루프에서 'integer' 선언 제거
#
import torch
import numpy as np

def format_real(val):
    """ Verilog-A 실수 포맷 (HSPICE 호환) """
    return f"{val:.16e}"

def get_params(model):
    """ 모델에서 모든 파라미터를 추출 """
    params = {}
    params['bz_'] = model.z_backbone[0].bias.data.cpu().numpy()
    params['Wz_'] = model.z_backbone[0].weight.data.cpu().numpy().flatten()
    params['bref_'] = model.head_ref.bias.data.cpu().numpy()[0]
    params['Wref_'] = model.head_ref.weight.data.cpu().numpy().flatten()
    params['bhr1_'] = model.head_hr[0].bias.data.cpu().numpy()
    params['Whr1_'] = model.head_hr[0].weight.data.cpu().numpy().flatten()
    params['bhr2_'] = model.head_hr[2].bias.data.cpu().numpy()[0]
    params['Whr2_'] = model.head_hr[2].weight.data.cpu().numpy().flatten()
    params['blr1_'] = model.head_lr[0].bias.data.cpu().numpy()
    params['Wlr1_'] = model.head_lr[0].weight.data.cpu().numpy().flatten()
    params['blr2_'] = model.head_lr[2].bias.data.cpu().numpy()[0]
    params['Wlr2_'] = model.head_lr[2].weight.data.cpu().numpy().flatten()
    return params

def export_veriloga_hspice_split(
    model, scX, R_ref_minmax, ny,
    out_base="narx_rram",
    module_name="narx_rram",
    Ts=1e-9, time_step=1e-9,
    init_w=1, warmup_steps=40,
    r_clip_min=100.0, r_clip_max=1.0e8,
    v_filter=0.01
):
    """
    [수정] HSPICE 호환 3-file Verilog-A 모델(셸, assign, math)을 모두 생성.
    """
    print(f"[Export] Generating 3 Verilog-A model files ({out_base}*)...")
    
    # 1. 파라미터 추출
    model.eval()
    params = get_params(model)
    in_dim = len(scX.mean_)
    z_dim = len(params['bz_'])
    hr_lr_in_dim = model.head_hr[0].in_features
    hr_lr_hid_dim = model.head_hr[0].out_features
    
    # 3개 파일 경로 정의
    shell_path = f"{out_base}.vams"
    assign_path = f"{out_base}_assign.vams"
    math_path = f"{out_base}_unrolled_math.inc"

    # ==================================================================
    # 1. 메인 셸(Shell) 파일 생성 (narx_rram.vams)
    # [수정] for 루프 내 integer 선언 제거
    # ==================================================================
    try:
        with open(shell_path, "w") as f:
            f.write(f"""// ----------------------------------------------------
//  NARX-based RRAM compact model (HSPICE-safe 3-file version)
//  - Main Shell File (auto-generated)
//  - numeric assignments: `include "{module_name}_assign.vams"
//  - unrolled math:       `include "{module_name}_unrolled_math.inc"
// ----------------------------------------------------
`include "constants.vams"
`include "disciplines.vams"

module {module_name}(g, s);
  inout g, s;
  electrical g, s;

  // ================== Parameters ==================
  parameter real Ts          = {format_real(Ts)};
  parameter integer N_warmup = {warmup_steps};
  parameter integer INIT_W   = {init_w};
  parameter real V_FILTER    = {format_real(v_filter)};
  parameter real Rref_min    = {format_real(R_ref_minmax[0])};
  parameter real Rref_max    = {format_real(R_ref_minmax[1])};
  
  parameter real tau_slew    = 2e-09 from [0:inf);
  parameter real R_clip_min  = {format_real(r_clip_min)};
  parameter real R_clip_max  = {format_real(r_clip_max)};
  parameter integer DBG      = 0;
  parameter real time_step   = {format_real(time_step)}; 

  // ================== Sizes (모델 구조) ==================
  integer ny, z_dim, in_dim;
  integer hr_lr_in_dim, hr_lr_hid_dim;
  integer i, m; // for-loop 인덱스

  // ================== States / temps ==================
  real V_prev_raw;
  real dV, dir, rate;
  real I_hold;
  real I_pred;
  integer k_sample_total;
  integer k_eff;
  real R_last, G_last;

  // ================== Input / feature buffers ==================
  real x_raw[0:{in_dim-1}];
  real x_scl[0:{in_dim-1}];

  // ================== Scaler buffers (값은 assign에서 대입) ==================
  real mean_[0:{in_dim-1}];
  real scale_[0:{in_dim-1}];

  // ================== Weights / Biases (값은 assign에서 대입) ==================
  real bz_[0:{z_dim-1}];
  real Wz_[0:{(z_dim*in_dim)-1}];
  real Wref_[0:{z_dim-1}];
  real bref_;
  real Whr1_[0:{(hr_lr_hid_dim*hr_lr_in_dim)-1}];
  real bhr1_[0:{hr_lr_hid_dim-1}];
  real Whr2_[0:{hr_lr_hid_dim-1}];
  real bhr2_;
  real Wlr1_[0:{(hr_lr_hid_dim*hr_lr_in_dim)-1}];
  real blr1_[0:{hr_lr_hid_dim-1}];
  real Wlr2_[0:{hr_lr_hid_dim-1}];
  real blr2_;

  // ================== Network intermediates (unrolled가 써 넣음) ==================
  real z[0:{z_dim-1}];
  real hr_h[0:{hr_lr_hid_dim-1}];
  real lr_h[0:{hr_lr_hid_dim-1}];

  // ================== Digital history ==================
  real W_hist[0:{ny-1}];

  // ================== Outputs / temps ==================
  real r_ref_logit, v_scl, log_hr, log_lr;
  real hr_val, lr_val, alpha, R_pred;
  real W_next, Vgs;
  real alpha_slew;

  // --- 가중치/바이어스/스케일러 로드 (assign 키워드 없음) ---
  `include "{module_name}_assign.vams"

  analog begin
    $bound_step(Ts);

    @(initial_step("tran")) begin
      ny            = {ny};
      z_dim         = {z_dim};
      in_dim        = {in_dim};
      hr_lr_in_dim  = {hr_lr_in_dim};
      hr_lr_hid_dim = {hr_lr_hid_dim};

      V_prev_raw = V(g, s);
      I_hold     = 0.0;
      I_pred     = 0.0;
      R_last     = R_clip_min;
      G_last     = 1.0 / R_last;
      k_sample_total = 0;
      k_eff      = 0;

      // [수정] for (integer i=0...) -> for (i=0...)
      for (i = 0; i <= ny-1; i = i+1) W_hist[i] = INIT_W;

      if (DBG) $strobe("{module_name}: loaded (ny=%0d, z_dim=%0d, in_dim=%0d, V_FILTER=%g, tau_slew=%g)", 
                        ny, z_dim, in_dim, V_FILTER, tau_slew);
    end

    // 출력: ZOH(+LPF)
    I(g, s) <+ I_hold;

    // 메인 업데이트
    @(timer(0, Ts)) begin
      Vgs = V(g, s);

      dV  = Vgs - V_prev_raw;
      dir = (dV > 0) ?  1 : ((dV < 0) ? -1 : 0);
      rate = abs(dV);

      alpha_slew = (tau_slew <= 0.0) ? 1.0 : (Ts / (tau_slew + Ts));
      if (alpha_slew > 1.0) alpha_slew = 1.0;
      else if (alpha_slew < 0.0) alpha_slew = 0.0;

      if (abs(Vgs) < V_FILTER) begin
        I_pred = Vgs * G_last;
        I_hold = I_hold + alpha_slew * (I_pred - I_hold);

        if (DBG) $strobe("SKIP  t=%g  V=%g  |V|<V_FILTER  -> I_read=%g (G_last=%g, k_eff=%0d)",
                          $abstime, Vgs, I_pred, G_last, k_eff);
        
        V_prev_raw = Vgs;
      end
      else begin
        x_raw[0] = Vgs;
        x_raw[1] = dir;
        x_raw[2] = rate;
        // [수정] for (integer i=0...) -> for (i=0...)
        for (i = 0; i <= ny-1; i = i+1) x_raw[i+3] = W_hist[i];

        // [수정] for (integer i=0...) -> for (i=0...)
        for (i = 0; i <= in_dim-1; i = i+1)
          x_scl[i] = (x_raw[i] - mean_[i]) / scale_[i];

        v_scl = x_scl[0];

        `include "{module_name}_unrolled_math.inc"

        hr_val = exp(log_hr);
        lr_val = exp(log_lr);

        alpha = (r_ref_logit - Rref_min) / (Rref_max - Rref_min);
        if (alpha < 0.0) alpha = 0.0;
        else if (alpha > 1.0) alpha = 1.0;

        R_pred = alpha * hr_val + (1.0 - alpha) * lr_val;
        if (R_pred < R_clip_min) R_pred = R_clip_min;
        if (R_pred > R_clip_max) R_pred = R_clip_max;

        I_pred = (R_pred != 0.0) ? (Vgs / R_pred) : 0.0;

        if (DBG) $strobe("INT t=%g | r_ref=%g log_hr=%g log_lr=%g",
                          $abstime, r_ref_logit, log_hr, log_lr);

        I_hold = I_hold + alpha_slew * (I_pred - I_hold);

        R_last = R_pred;
        G_last = 1.0 / R_last;

        W_next = (r_ref_logit > 0) ? 1 : 0;
        
        if (k_eff >= N_warmup) begin
          // [수정] for (integer m=...) -> for (m=...)
          for (m = ny-1; m >= 1; m = m-1)
            W_hist[m] = W_hist[m-1];
          W_hist[0] = W_next;
        end
        k_eff = k_eff + 1;

        if (DBG) $strobe("VALID t=%g V=%g dV=%g dir=%g rate=%g | alpha=%g  R=%g  I=%g  W0=%g  k_eff=%0d",
                          $abstime, Vgs, dV, dir, rate, alpha, R_pred, I_pred, W_hist[0], k_eff);

        V_prev_raw = Vgs;
      end

      k_sample_total = k_sample_total + 1;
    end
  end
endmodule
""")
        print(f"[Export] Successfully wrote main shell file: {shell_path}")
    except Exception as e:
        print(f"[Export] ERROR writing main shell file: {e}")
        return

    # ==================================================================
    # 2. _assign.vams (가중치, 바이어스, 스케일러) 파일 생성
    # [수정] 'assign' 키워드 제거
    # ==================================================================
    try:
        with open(assign_path, "w") as f:
            f.write("// Verilog-A numeric assignments (auto-generated)\n")
            f.write(f"// Included by: {module_name}.vams\n")
            f.write("`include \"constants.vams\"\n\n")

            f.write("// --- Scaler (mean/scale) ---\n")
            for i in range(in_dim):
                f.write(f"mean_[{i}] = {format_real(scX.mean_[i])};\n")
                f.write(f"scale_[{i}] = {format_real(scX.scale_[i])};\n")
            
            f.write("\n// --- Z_backbone (bz_, Wz_) ---\n")
            for i in range(z_dim):
                f.write(f"bz_[{i}] = {format_real(params['bz_'][i])};\n")
            for i in range(z_dim * in_dim):
                f.write(f"Wz_[{i}] = {format_real(params['Wz_'][i])};\n")

            f.write("\n// --- Head_ref (bref_, Wref_) ---\n")
            f.write(f"bref_ = {format_real(params['bref_'])};\n")
            for i in range(z_dim):
                f.write(f"Wref_[{i}] = {format_real(params['Wref_'][i])};\n")

            f.write("\n// --- Head_hr ---\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"bhr1_[{i}] = {format_real(params['bhr1_'][i])};\n")
            for i in range(hr_lr_hid_dim * hr_lr_in_dim):
                f.write(f"Whr1_[{i}] = {format_real(params['Whr1_'][i])};\n")
            f.write(f"bhr2_ = {format_real(params['bhr2_'])};\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"Whr2_[{i}] = {format_real(params['Whr2_'][i])};\n")
                
            f.write("\n// --- Head_lr ---\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"blr1_[{i}] = {format_real(params['blr1_'][i])};\n")
            for i in range(hr_lr_hid_dim * hr_lr_in_dim):
                f.write(f"Wlr1_[{i}] = {format_real(params['Wlr1_'][i])};\n")
            f.write(f"blr2_ = {format_real(params['blr2_'])};\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"Wlr2_[{i}] = {format_real(params['Wlr2_'][i])};\n")
                
            f.write("\n// --- End of assignments ---\n")
        print(f"[Export] Successfully wrote assign file: {assign_path}")
    except Exception as e:
        print(f"[Export] ERROR writing assign file: {e}")
        return

    # ==================================================================
    # 3. _unrolled_math.inc (계산식) 파일 생성 (변경 없음)
    # ==================================================================
    try:
        with open(math_path, "w") as f:
            f.write("// Verilog-A unrolled math (auto-generated)\n")
            f.write(f"// Included by: {module_name}.vams\n")
            f.write("// [!!] DO NOT add 'real' declarations here [!!]\n\n")

            f.write("// --- A. Z_backbone (x_scl -> z) ---\n")
            for i in range(z_dim):
                line = f"z[{i}] = bz_[{i}]"
                for j in range(in_dim):
                    line += f" + Wz_[{i * in_dim + j}]*x_scl[{j}]"
                line += ";\n"
                f.write(line)
                f.write(f"z[{i}] = tanh(z[{i}]);\n")
            
            f.write("\n// --- B. Head_ref (z -> r_ref_logit) ---\n")
            line = "r_ref_logit = bref_"
            for i in range(z_dim):
                line += f" + Wref_[{i}]*z[{i}]"
            line += ";\n"
            f.write(line)

            f.write("\n// --- C. Head_hr (z, v_scl -> log_hr) ---\n")
            for i in range(hr_lr_hid_dim):
                line = f"hr_h[{i}] = bhr1_[{i}]"
                for j in range(z_dim): 
                    line += f" + Whr1_[{i * hr_lr_in_dim + j}]*z[{j}]"
                line += f" + Whr1_[{i * hr_lr_in_dim + z_dim}]*v_scl";
                line += ";\n"
                f.write(line)
                f.write(f"hr_h[{i}] = tanh(hr_h[{i}]);\n")
            line = "log_hr = bhr2_"
            for i in range(hr_lr_hid_dim):
                line += f" + Whr2_[{i}]*hr_h[{i}]"
            line += ";\n"
            f.write(line)
            
            f.write("\n// --- D. Head_lr (z, v_scl -> log_lr) ---\n")
            for i in range(hr_lr_hid_dim):
                line = f"lr_h[{i}] = blr1_[{i}]"
                for j in range(z_dim):
                    line += f" + Wlr1_[{i * hr_lr_in_dim + j}]*z[{j}]"
                line += f" + Wlr1_[{i * hr_lr_in_dim + z_dim}]*v_scl";
                line += ";\n"
                f.write(line)
                f.write(f"lr_h[{i}] = tanh(lr_h[{i}]);\n")
            line = "log_lr = blr2_"
            for i in range(hr_lr_hid_dim):
                line += f" + Wlr2_[{i}]*lr_h[{i}]"
            line += ";\n"
            f.write(line)
            
            f.write("\n// --- End of math ---\n")
        print(f"[Export] Successfully wrote unrolled math file: {math_path}")

    except Exception as e:
        print(f"[Export] ERROR writing unrolled math file: {e}")
        return

    print(f"[Export] All 3 Verilog-A files ({shell_path}, {assign_path}, {math_path}) generated successfully.")