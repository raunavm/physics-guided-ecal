# Stage B (LBFGS) with physics fixed

function _lb_shapes(nn0)
    return (
        l1W = size(nn0.layer_1.weight), l1b = size(nn0.layer_1.bias),
        l2W = size(nn0.layer_2.weight), l2b = size(nn0.layer_2.bias),
        l3W = size(nn0.layer_3.weight), l3b = size(nn0.layer_3.bias),
        l4W = size(nn0.layer_4.weight), l4b = size(nn0.layer_4.bias),
    )
end

lb_pack_nn(nn, LB_SHAPES) = vcat(
    vec(nn.layer_1.weight), vec(nn.layer_1.bias),
    vec(nn.layer_2.weight), vec(nn.layer_2.bias),
    vec(nn.layer_3.weight), vec(nn.layer_3.bias),
    vec(nn.layer_4.weight), vec(nn.layer_4.bias),
)

function lb_unpack_nn(x::AbstractVector{<:Real}, LB_SHAPES)
    i = 1
    n1W = prod(LB_SHAPES.l1W);  W1 = reshape(x[i:i+n1W-1], LB_SHAPES.l1W); i += n1W
    n1b = prod(LB_SHAPES.l1b);  b1 = reshape(x[i:i+n1b-1], LB_SHAPES.l1b); i += n1b
    n2W = prod(LB_SHAPES.l2W);  W2 = reshape(x[i:i+n2W-1], LB_SHAPES.l2W); i += n2W
    n2b = prod(LB_SHAPES.l2b);  b2 = reshape(x[i:i+n2b-1], LB_SHAPES.l2b); i += n2b
    n3W = prod(LB_SHAPES.l3W);  W3 = reshape(x[i:i+n3W-1], LB_SHAPES.l3W); i += n3W
    n3b = prod(LB_SHAPES.l3b);  b3 = reshape(x[i:i+n3b-1], LB_SHAPES.l3b); i += n3b
    n4W = prod(LB_SHAPES.l4W);  W4 = reshape(x[i:i+n4W-1], LB_SHAPES.l4W); i += n4W
    n4b = prod(LB_SHAPES.l4b);  b4 = reshape(x[i:i+n4b-1], LB_SHAPES.l4b); i += n4b
    return (layer_1=(weight=W1, bias=b1), layer_2=(weight=W2, bias=b2),
            layer_3=(weight=W3, bias=b3), layer_4=(weight=W4, bias=b4))
end

function lb_rhs(u::AbstractVector, nn_params, s::Real, t_raw::Real, PHYS_FIXED)
    λ = lambda_gamma_slope(s, t_raw, PHYS_FIXED)
    β = beta_effective(PHYS_FIXED)
    urelu = @. max(u, 0.0); m_u = mean(u)
    if β > BETA_TOL
        lap = LAPL * u; agg = deg .* u .- lap
        N = length(u); F = reshape(vcat(u', urelu', agg', fill(s, N)', fill(m_u, N)'), 5, N)
        corr = vec(gnn_apply(gnn, nn_params, st, F))
        return @. λ*u - β*lap + corr
    else
        agg = deg .* u
        N = length(u); F = reshape(vcat(u', urelu', agg', fill(s, N)', fill(m_u, N)'), 5, N)
        corr = vec(gnn_apply(gnn, nn_params, st, F))
        return @. λ*u + corr
    end
end

function lb_residual_loss_vec(xvec::AbstractVector{<:Real}, events::Vector{Int}, LB_SHAPES, PHYS_FIXED)::Float64
    nn_params = lb_unpack_nn(xvec, LB_SHAPES)
    @inline softarg_pos(x::Vector{Float64}, pos::Vector{Float64}, τ::Real) = begin
        z = (x .- maximum(x)) ./ τ
        w = exp.(z); w = w ./ (sum(w) + EPS)
        sum(w .* pos)
    end

    totalsum = 0.0; cnt = 0
    @inbounds for e in events
        Xt = Xn[e]; T, _ = size(Xt); ss = TS_CACHE[e]; tsraw = T_CENTERS_RAW
        t_rng = 1:(T-1); isempty(t_rng) && continue

        per = map(t_rng) do t
            u      = vec(@view Xt[t,   :])
            u_next = vec(@view Xt[t+1, :])
            Δs     = ss[t+1] - ss[t]

            du_m   = Δs .* lb_rhs(u, nn_params, ss[t], tsraw[t], PHYS_FIXED)
            upred  = u .+ du_m

            E_true_next = energy_from_normalized(u_next, μ, σ)
            upost, _    = postprocess_step(upred; μ=μ, σ=σ, target_E=E_true_next, apply_micro=false)

            ksub = PIX_SUBSAMPLE_REF[]
            if ksub > 0 && ksub < length(u)
                idxs = _subidx_for_event_layer(e, t, ksub)
                if idxs === nothing
                    du_fd = u_next .- u
                    du_pp = upost   .- u
                else
                    du_fd = (u_next .- u)[idxs]
                    du_pp = (upost   .- u)[idxs]
                end
            else
                du_fd = u_next .- u
                du_pp = upost   .- u
            end
            denom = USE_STEP_SCALING ? max((mean(abs.(u)) + mean(abs.(u_next)))*0.5, 1e-8) : 1.0
            resid = mean(((du_fd .- du_pp) ./ denom).^2)
            (resid = resid, E_true_next = E_true_next, spos = ss[t+1])
        end

        step_losses = getfield.(per, :resid)
        e_pred_vec  = getfield.(per, :E_true_next)
        e_true_vec  = e_pred_vec
        pos         = getfield.(per, :spos)

        tmax_pred = softarg_pos(e_pred_vec, pos, SOFTARG_T)
        tmax_true = softarg_pos(e_true_vec, pos, SOFTARG_T)
        loss_tmax = (tmax_pred - tmax_true)^2
        Etot_pred = sum(e_pred_vec); Etot_true = sum(e_true_vec)
        loss_etot = ((Etot_pred - Etot_true)/(abs(Etot_true)+1e-8))^2

        totalsum += mean(step_losses) + LAMBDA_TMAX_REF[]*loss_tmax + LAMBDA_ETOTAL_REF[]*loss_etot
        cnt += 1
    end
    return totalsum / max(cnt, 1)
end

function train_stage_b!()
    set_mode!(:full)
    set_beta_scale!(BETA_STAGEB_SCALE)
    set_beta_clamp!(BETA_STAGEB_MAX)

    PHYS_FIXED = Zygote.ignore() do deepcopy(theta_best[].phys) end
    LB_SHAPES = _lb_shapes(theta_best[].nn)
    x0 = lb_pack_nn(theta_best[].nn, LB_SHAPES)

    lb_loss_vec = x -> lb_residual_loss_vec(x, EVAL_IDX_CACHED, LB_SHAPES, PHYS_FIXED)
    optfun = Optimization.OptimizationFunction((x, p) -> lb_loss_vec(x), Optimization.AutoZygote())
    prob   = Optimization.OptimizationProblem(optfun, x0)

    lb_iters       = Ref(0)
    lb_best_fx     = Ref(lb_loss_vec(x0))
    lb_best_x      = Ref(copy(x0))
    lb_no_improve  = Ref(0)
    lb_patience    = 100
    lb_rtol        = 1e-7

    lbfgs_cb = function (state, args...)
        lb_iters[] += 1
        x_now = state.u
        fx    = lb_loss_vec(x_now)
        if isfinite(fx) && (fx + lb_rtol * abs(lb_best_fx[]) < lb_best_fx[])
            lb_best_fx[]   = fx
            lb_best_x[]    = copy(x_now)
            lb_no_improve[] = 0
        else
            lb_no_improve[] += 1
        end
        if (lb_iters[] == 1) || (lb_iters[] % 10 == 0)
            @info "stageB/lbfgs" iter=lb_iters[] eval_resid=fx
        end
        return (!isfinite(fx)) || (lb_no_improve[] >= lb_patience)
    end

    sol = Optimization.solve(
        prob,
        OptimizationOptimJL.LBFGS();
        maxiters    = 300,
        callback    = lbfgs_cb,
        show_trace  = false,
        store_trace = false,
    )

    nn_star = lb_unpack_nn(lb_best_x[], LB_SHAPES)
    global theta_star = ComponentVector((phys = PHYS_FIXED, nn = nn_star))

    ret = (sol isa SciMLBase.AbstractSciMLSolution) ? sol.retcode : :ok
    @info "Stage B finished" best_eval=lb_best_fx[] iters=lb_iters[] retcode=ret
end
