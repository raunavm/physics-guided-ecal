# Defaults & hyperparams (as Refs so you can mutate at runtime)

if !@isdefined H5                ; const H5 = "data/dataset_2_1.hdf5"      ; end
if !@isdefined CC_XML            ; const CC_XML = ""                        ; end
if !@isdefined MAX_EVENTS        ; const MAX_EVENTS = nothing               ; end
if !@isdefined RNG               ; const RNG = StableRNGs.StableRNG(0x8af)  ; end

if !@isdefined BETA_STAGEA_SCALE      ; const BETA_STAGEA_SCALE      = 0.0      ; end
if !@isdefined BETA_STAGEB_SCALE      ; const BETA_STAGEB_SCALE      = 1.0      ; end
if !@isdefined BETA_STAGEB_MAX        ; const BETA_STAGEB_MAX        = 0.0      ; end
if !@isdefined BETA_FORWARD_SCALE_REF ; const BETA_FORWARD_SCALE_REF = Ref(BETA_STAGEA_SCALE); end
if !@isdefined BETA_CLAMP_MAX_REF     ; const BETA_CLAMP_MAX_REF     = Ref(0.0) ; end

if !@isdefined TMIN ; const TMIN = 1e-3 ; end
if !@isdefined EPS  ; const EPS  = 1e-9 ; end
if !@isdefined BETA_TOL     ; const BETA_TOL     = 1e-7  ; end
if !@isdefined ENERGY_EPS   ; const ENERGY_EPS   = 1e-10 ; end

if !@isdefined CURRENT_MODE ; const CURRENT_MODE = Ref(:full)     ; end
if !@isdefined ENERGY_TARGET; const ENERGY_TARGET= Ref(:conserve) ; end
if !@isdefined MICRO_ALPHA_REF      ; const MICRO_ALPHA_REF       = Ref(0.25) ; end
if !@isdefined MICRO_DIFF_STEPS_REF ; const MICRO_DIFF_STEPS_REF  = Ref(1)    ; end
if !@isdefined MICRO_DIFF_ON        ; const MICRO_DIFF_ON         = Ref(false); end
if !@isdefined SMOKE_MODE           ; const SMOKE_MODE            = Ref(false); end

if !@isdefined EPOCHS_ADAM_REF        ; const EPOCHS_ADAM_REF    = Ref(1200) ; end
if !@isdefined ADAM_LR_REF            ; const ADAM_LR_REF        = Ref(5e-4) ; end
if !@isdefined BATCH_BASE_REF         ; const BATCH_BASE_REF     = Ref(8)    ; end
if !@isdefined FAST_BATCH_MULT_REF    ; const FAST_BATCH_MULT_REF= Ref(8)    ; end
if !@isdefined PIX_SUBSAMPLE_REF      ; const PIX_SUBSAMPLE_REF  = Ref(0)    ; end
if !@isdefined TIME_SAMPLES_REF       ; const TIME_SAMPLES_REF   = Ref(8)    ; end
if !@isdefined BETA_FREEZE_STEPS_REF  ; const BETA_FREEZE_STEPS_REF=Ref(40)  ; end
if !@isdefined WARMUP_STEPS_REF       ; const WARMUP_STEPS_REF   = Ref(5)    ; end
if !@isdefined EVAL_EVERY_REF         ; const EVAL_EVERY_REF     = Ref(50)   ; end
if !@isdefined PATIENCE_REF           ; const PATIENCE_REF       = Ref(200)  ; end

if !@isdefined RESIDUAL_SCALE  ; const RESIDUAL_SCALE  = 1.0 ; end
if !@isdefined USE_STEP_SCALING; const USE_STEP_SCALING= false ; end
if !@isdefined LAMBDA_TMAX_REF ; const LAMBDA_TMAX_REF = Ref(1.0) ; end
if !@isdefined LAMBDA_ETOTAL_REF;const LAMBDA_ETOTAL_REF=Ref(0.25); end
if !@isdefined SOFTARG_T       ; const SOFTARG_T       = 0.05 ; end

if !@isdefined CLIP_THRESH ; const CLIP_THRESH = 0.35 ; end
if !@isdefined BOOST_GRADS ; const BOOST_GRADS = 1.25 ; end

if !@isdefined EVAL_EVENTS_MAX ; const EVAL_EVENTS_MAX = 10 ; end

# Global state produced during prepare! / training (kept in module for simplicity)
# Data / normalization
if !@isdefined X        ; global X        ; end
if !@isdefined Xn       ; global Xn       ; end
if !@isdefined μ        ; global μ        ; end
if !@isdefined σ        ; global σ        ; end
if !@isdefined train_idx; global train_idx; end
if !@isdefined val_idx  ; global val_idx  ; end
if !@isdefined H        ; global H        ; end
if !@isdefined W        ; global W        ; end

# Depth/time grids
if !@isdefined T_CENTERS_RAW ; global T_CENTERS_RAW ; end
if !@isdefined S_CENTERS     ; global S_CENTERS     ; end
if !@isdefined TS_CACHE      ; global TS_CACHE      ; end

# Graph/Laplacian
if !@isdefined nbrs ; global nbrs ; end
if !@isdefined deg  ; global deg  ; end
if !@isdefined LAPL ; global LAPL ; end
if !@isdefined NPIX ; global NPIX ; end
if !@isdefined LAPL_RHO_REF ; const LAPL_RHO_REF = Ref(0.0) ; end

# Model params / states
if !@isdefined gnn  ; global gnn  ; end
if !@isdefined p_nn ; global p_nn ; end
if !@isdefined st   ; global st   ; end
if !@isdefined theta; global theta; end

# Training traces / caches
if !@isdefined EVAL_IDX_CACHED; global EVAL_IDX_CACHED; end
if !@isdefined BSZ; global BSZ; end
if !@isdefined idx_buf; global idx_buf; end
if !@isdefined ALL_IF_NO_SPLIT; global ALL_IF_NO_SPLIT; end
if !@isdefined optstate; global optstate; end
if !@isdefined train_trace; global train_trace; end
if !@isdefined eval_trace ; global eval_trace ; end
if !@isdefined theta_best ; global theta_best ; end
if !@isdefined theta_star ; global theta_star ; end
if !@isdefined best_eval  ; global best_eval  ; end
if !@isdefined best_eval_step; global best_eval_step; end
