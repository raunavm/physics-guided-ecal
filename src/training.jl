# Pixel subsampling helper
function _subidx_for_event_layer(e::Int, t::Int, k::Int)
    if k <= 0 || k >= NPIX; return nothing; end
    rng = StableRNGs.StableRNG(0xBEEF + 97*e + t)
    idxs = collect(1:NPIX)
    Random.shuffle!(rng, idxs)
    return @view idxs[1:k]
end

# Loss for a subset of events (no mutation, AD-safe)
function residual_loss_subset(theta, events::Vector{Int}; Xn_local = Xn, gnn_local=gnn, st_local=st)
    @inline softarg_pos(x::Vector{Float64}, pos::Vector{Float64}, τ::Real) = begin
        z = (x .- maximum(x)) ./ τ
        w = exp.(z); w = w ./ (sum(w) + EPS)
        sum(w .* pos)
    end
    per_event_losses = map(events) do e
        Xt   = Xn_local[e]
        T, _ = size(Xt)
        ss    = TS_CACHE[e]
        tsraw = T_CENTERS_RAW

        t_idx = Zygote.ignore() do
            if T <= 2
                Int[]
            elseif TIME_SAMPLES_REF[] <= 0 || (T-1) <= TIME_SAMPLES_REF[]
                collect(1:(T-1))
            else
                k = TIME_SAMPLES_REF[]
                collect(rand(StableRNGs.StableRNG(0xBEEF + e), 1:(T-1), k))
            end
        end
        isempty(t_idx) && return 0.0

        per = map(t_idx) do t
            u      = vec(Xt[t,   :])
            u_next = vec(Xt[t+1, :])
            Δs     = ss[t+1] - ss[t]

            du_m   = Δs .* f_eval(u, theta, ss[t], tsraw[t], deg, gnn_local, st_local)
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
        e_vec       = getfield.(per, :E_true_next)
        pos         = getfield.(per, :spos)

        tmax_pred = softarg_pos(e_vec, pos, SOFTARG_T)
        tmax_true = softarg_pos(e_vec, pos, SOFTARG_T)
        loss_tmax = (tmax_pred - tmax_true)^2
        Etot_pred = sum(e_vec);  Etot_true = sum(e_vec)
        loss_etot = ((Etot_pred - Etot_true)/(abs(Etot_true)+1e-8))^2

        mean(step_losses) + LAMBDA_TMAX_REF[]*loss_tmax + LAMBDA_ETOTAL_REF[]*loss_etot
    end

    total = isempty(per_event_losses) ? 0.0 : mean(per_event_losses)
    total += 1e-4 * (beta_effective(theta.phys) - 0.02)^2
    return RESIDUAL_SCALE * total
end

function _ensure_buffers!()
    global BSZ
    if !@isdefined(BSZ)
        total = isempty(train_idx) ? length(X) : length(train_idx)
        BSZ   = max(1, min(FAST_BATCH_MULT_REF[] * BATCH_BASE_REF[], total))
    end
    if !@isdefined(idx_buf) || length(idx_buf) != BSZ
        global idx_buf = Vector{Int}(undef, BSZ)
    end
end

function train_stage_a!()
    set_mode!(:nn_first)
    set_beta_scale!(BETA_STAGEA_SCALE)
    set_beta_clamp!(0.0)

    local adam = Optimisers.OptimiserChain(Optimisers.ClipNorm(CLIP_THRESH), Optimisers.Adam(ADAM_LR_REF[]))
    global optstate = Optimisers.setup(adam, theta)
    global train_trace = Float64[]
    global eval_trace  = Float64[]
    global best_eval   = Ref(Inf)
    global best_eval_step = Ref(0)
    global theta_best  = Ref(deepcopy(theta))
    _ensure_buffers!()
    global ALL_IF_NO_SPLIT = isempty(train_idx)

    if SMOKE_MODE[]
        @assert LAPL_RHO_REF[] > 0 "spectral radius must be positive"
        let u = copy(vec(Xn[1][1,:])), tgt = 123.45
            up, _ = postprocess_step(u; μ=μ, σ=σ, target_E=tgt, apply_micro=false)
            @assert abs(energy_from_normalized(up, μ, σ) - tgt) < 1e-6 "energy not preserved"
        end
        sm = residual_loss_subset(theta, (isempty(train_idx) ? [1] : [train_idx[1]])) / max(RESIDUAL_SCALE, 1e-12)
        @assert isfinite(sm) "smoke residual not finite"
        println("smoke tests passed.")
    end

    for step in 1:EPOCHS_ADAM_REF[]
        local idx_local::Vector{Int}
        if ALL_IF_NO_SPLIT
            idx_local = (length(idx_buf) == length(X)) ? idx_buf : (resize!(idx_buf, length(X)); idx_buf)
            @inbounds for i in eachindex(idx_local); idx_local[i] = i; end
        else
            @inbounds for i in eachindex(idx_buf)
                idx_buf[i] = train_idx[rand(RNG, 1:length(train_idx))]
            end
            idx_local = idx_buf
        end

        local θ = theta
        loss_scaled, back = Zygote.pullback(th -> residual_loss_subset(th, idx_local), θ)
        g = first(back(1.0))

        if hasproperty(g, :phys) && step <= BETA_FREEZE_STEPS_REF[]
            g = ComponentVector((phys = zero(g.phys), nn = g.nn))
        end
        g = (BOOST_GRADS == 1.0) ? g : (BOOST_GRADS .* g)

        g_norm = sqrt(sum(abs2, ComponentArrays.getdata(g)) + 1e-12)
        local new_optstate, θ_new = Optimisers.update(optstate, θ, g)
        global optstate = new_optstate
        global theta    = θ_new

        raw_loss = loss_scaled / max(RESIDUAL_SCALE, 1e-12)
        push!(train_trace, raw_loss)

        if (step % EVAL_EVERY_REF[] == 0) && (step > WARMUP_STEPS_REF[])
            l_eval = residual_loss_subset(theta, EVAL_IDX_CACHED) / max(RESIDUAL_SCALE,1e-12)
            push!(eval_trace, l_eval)
            if l_eval < best_eval[]
                best_eval[]      = l_eval
                best_eval_step[] = step
                theta_best[]     = deepcopy(theta)
            end
            @info "stageA/adam" iter=step train_resid=raw_loss eval_resid=l_eval grad_norm=g_norm
            if (step - best_eval_step[]) >= PATIENCE_REF[]
                @info "stageA early stop (patience)" iter=step
                break
            end
        elseif step == 1
            @info "stageA/adam" iter=step train_resid=raw_loss grad_norm=g_norm
        end
    end

    @info "Stage A finished" best_eval=best_eval[] best_step=best_eval_step[]
end
