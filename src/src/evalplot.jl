# Curves & metrics & saving

next_energy_target(u, s, t_raw, Δs, th) = begin
    E_base = energy_from_normalized(u, μ, σ)
    λ      = lambda_gamma_slope(s, t_raw, th.phys)
    E_base * exp(λ * Δs)
end

function _curve(th, Xt, ss, tsraw; post::Bool, micro::Bool, target::Symbol=:data)
    e_true = Float64[]; e_pred = Float64[]; s = Float64[]
    T, _ = size(Xt)
    for t in 1:(T-1)
        u  = vec(@view Xt[t,   :])
        v  = vec(@view Xt[t+1, :])
        Δs = ss[t+1] - ss[t]

        if post
            E_tgt = (target === :data) ?
                energy_from_normalized(v, μ, σ) :
                next_energy_target(u, ss[t], tsraw[t], Δs, th)
            up = step_predict(u, th, ss[t], tsraw[t], Δs;
                              target_E=E_tgt, apply_projection=true, apply_microdiff=micro)
        else
            du = Δs .* f_eval(u, th, ss[t], tsraw[t], deg, gnn, st)
            up = micro ? micro_diffuse(u .+ du) : (u .+ du)
        end

        push!(e_true, energy_from_normalized(v,  μ, σ))
        push!(e_pred, energy_from_normalized(up, μ, σ))
        push!(s, ss[t+1])
    end
    return e_true, e_pred, s
end

function _metrics(e_true::Vector{Float64}, e_pred::Vector{Float64}, s::Vector{Float64})
    err  = e_pred .- e_true
    rel  = err ./ (abs.(e_true) .+ 1e-12)
    mape = 100 * mean(abs.(rel))
    rmse = sqrt(mean(err .^ 2))
    r2   = 1 .- (sum((e_true .- e_pred).^2) / (sum((e_true .- mean(e_true)).^2) + 1e-12))
    i_true = argmax(e_true); i_pred = argmax(e_pred)
    peak_s_err     = s[i_pred] - s[i_true]
    peak_E_rel_err = (e_pred[i_true] - e_true[i_true]) / (abs(e_true[i_true]) + 1e-12)
    etot_rel       = (sum(e_pred) - sum(e_true)) / (abs(sum(e_true)) + 1e-12)
    (; r2, rmse, mape, peak_s_err, peak_E_rel_err, etot_rel, i_true, i_pred)
end

function evaluate_and_save!()
    th = @isdefined(theta_star) ? theta_star : (@isdefined(theta_best) ? theta_best[] : theta)
    set_mode!(:full)

    ev = (isdefined(Main, :EVAL_IDX_CACHED) && !isempty(EVAL_IDX_CACHED)) ? EVAL_IDX_CACHED[1] : 1
    ev = clamp(ev, 1, length(Xn))
    Xt = Xn[ev]; ss = TS_CACHE[ev]; tsraw = T_CENTERS_RAW

    # Plot settings
    APPLY_POST             = false
    MICRODIFF_IN_PLOTS     = false
    ENERGY_TARGET_IN_PLOTS = :data

    et, ep, spos = _curve(th, Xt, ss, tsraw; post=APPLY_POST, micro=MICRODIFF_IN_PLOTS, target=ENERGY_TARGET_IN_PLOTS)

    micro_tag  = MICRODIFF_IN_PLOTS ? "+micro" : ""
    target_tag = (ENERGY_TARGET_IN_PLOTS === :data) ? "data" : "phys"
    model_lbl  = APPLY_POST ? "Model (post→$(target_tag) target$(micro_tag))" :
                              (MICRODIFF_IN_PLOTS ? "Model (raw+micro)" : "Model (raw)")

    plt_curve = plot(spos, et; lw=3, label="Data (next layer)")
    plot!(plt_curve, spos, ep; lw=3, linestyle=:dash, label=model_lbl)
    xlabel!(plt_curve, "Normalized depth s"); ylabel!(plt_curve, "Total layer energy")
    title!(plt_curve, "Energy vs depth — event $ev"); plot!(plt_curve; grid=true)
    display(plt_curve)

    i_true = argmax(et)
    t_peak = i_true
    u_peak = vec(@view Xt[t_peak,   :])
    v_true = vec(@view Xt[t_peak+1, :])
    Δs_peak= ss[t_peak+1] - ss[t_peak]

    E_tgt  = (ENERGY_TARGET_IN_PLOTS === :data) ?
        energy_from_normalized(v_true, μ, σ) :
        next_energy_target(u_peak, ss[t_peak], tsraw[t_peak], Δs_peak, th)

    v_pred = step_predict(u_peak, th, ss[t_peak], tsraw[t_peak], Δs_peak;
                          target_E=E_tgt, apply_projection=true, apply_microdiff=MICRODIFF_IN_PLOTS)

    to_img(v) = reshape(denorm_global(v, μ, σ), H, W)
    true_img = to_img(v_true); pred_img = to_img(v_pred); diff_img = pred_img .- true_img

    vmin, vmax = 0.0, maximum(true_img)
    plt_hm = plot(layout=(1,3), size=(1200, 360))
    heatmap!(plt_hm[1], true_img; color=:viridis, clims=(vmin, vmax), axis=false, title="True (t=$(t_peak+1))")
    heatmap!(plt_hm[2], pred_img; color=:viridis, clims=(vmin, vmax), axis=false, title="Model (postprocessed)")
    m = maximum(abs.(diff_img)); divpal = cgrad(:RdBu, rev=true)
    heatmap!(plt_hm[3], diff_img; color=divpal, clims=(-m, m), axis=false, title="Diff (model − true)")
    display(plt_hm)

    M = _metrics(et, ep, spos)
    mean_abs_pix = mean(abs.(diff_img))
    max_abs_pix  = maximum(abs.(diff_img))
    eval_resid_val = residual_loss_subset(th, EVAL_IDX_CACHED) / max(RESIDUAL_SCALE,1e-12)

    println("\n=== EVALUATION SUMMARY ===")
    @printf("eval_resid = %.6f\n", eval_resid_val)
    @printf("Energy: R^2=%.6f  RMSE=%.3e  MAPE=%.4f%%  peakΔs=%.5f  relE@peak=%.5f  Etot_rel=%.5f\n",
            M.r2, M.rmse, M.mape, M.peak_s_err, M.peak_E_rel_err, M.etot_rel)
    @printf("Pixels@peak: mean|diff|=%.6f  max|diff|=%.6f\n", mean_abs_pix, max_abs_pix)
    println("==========================\n")

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    ckpt_path = "theta_star_$(timestamp).jld2"
    @save ckpt_path th μ σ H W
    @info "Saved checkpoint" path=ckpt_path

    metrics = Dict(
        "timestamp" => timestamp,
        "eval_resid" => eval_resid_val,
        "R2" => M.r2,
        "RMSE" => M.rmse,
        "MAPE_percent" => M.mape,
        "peak_timing_error" => M.peak_s_err,
        "peak_energy_rel_error" => M.peak_E_rel_err,
        "Etot_rel_error" => M.etot_rel,
        "mean_abs_pixel_diff_peak" => mean_abs_pix,
        "max_abs_pixel_diff_peak" => max_abs_pix,
        "laplacian_rho" => LAPL_RHO_REF[],
        "beta_effective" => beta_effective(th.phys),
        "EVAL_EVENTS" => EVAL_IDX_CACHED
    )
    metrics_path = "metrics_run_$(timestamp).json"
    open(metrics_path, "w") do io
        JSON.print(io, metrics, 4)
    end
    @info "Saved metrics" path=metrics_path
end
