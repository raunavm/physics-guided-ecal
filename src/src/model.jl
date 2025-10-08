# Small MLP residual (Lux)
function make_gnn(rng; h1=96, h2=96, h3=48)
    model = Lux.Chain(
        Lux.Dense(5,   h1, tanh),
        Lux.Dense(h1,  h2, tanh),
        Lux.Dense(h2,  h3, tanh),
        Lux.Dense(h3,  1)
    )
    p, st = Lux.setup(rng, model)
    return model, p, st
end
gnn_apply(model, p, st, F::AbstractMatrix) = first(Lux.apply(model, F, p, st))

# Physics seeding from data
function seed_phys_from_data(Xn, μ, σ, s_centers)::NamedTuple
    Em = zeros(length(s_centers))
    for e in 1:length(Xn)
        Xt = Xn[e]
        Em .+= [energy_from_normalized(vec(@view Xt[t,:]), μ, σ) for t in 1:size(Xt,1)]
    end
    Em ./= max(length(Xn),1); Em ./= max(sum(Em), 1e-12)
    function smooth_local_quadratic(y::Vector{Float64}, s::Vector{Float64}; win::Int=5)
        L = length(y); win≤1 && return copy(y)
        yy = similar(y); half = win ÷ 2
        for i in 1:L
            i0 = max(1, i-half); i1 = min(L, i+half)
            xs = s[i0:i1]; ys = y[i0:i1]
            X = hcat(ones(length(xs)), xs, xs.^2)
            β = X \ ys
            yy[i] = β[1] + β[2]*s[i] + β[3]*s[i]^2
        end
        yy
    end
    Es = smooth_local_quadratic(Em, s_centers; win=5)
    ipeak = argmax(Es); smax = s_centers[ipeak]
    g = Float64[]; smid = Float64[]
    for i in 1:length(Es)-1
        Ei, Ej = max(Es[i],1e-12), max(Es[i+1],1e-12)
        Δs = s_centers[i+1]-s_centers[i]
        push!(g, (log(Ej)-log(Ei))/max(Δs,1e-9))
        push!(smid, 0.5*(s_centers[i]+s_centers[i+1]))
    end
    Xls = hcat(1.0 ./ clamp.(smid, 1e-3, 1.0), ones(length(smid)))
    βls = Xls \ g
    α̂ = βls[1]
    a0   = max(1.03, 1.0 + α̂)
    τ0   = 1.0
    b0   = max((a0-1.0)/max(smax,0.15), 1e-3)
    toff = 0.35
    return (a=a0, b=b0, τ=τ0, t_off=toff)
end

function init_params(rng, p_nn)::ComponentVector
    ComponentVector((phys = (a_raw=0.0, b_raw=0.0, tau_raw=0.0, toff_raw=0.0, beta_raw=invsoftplus(0.02 + 1e-3)),
                     nn = p_nn))
end

beta_effective(phys; beta_eps=1e-3) = begin
    β = softplus(phys.beta_raw) - beta_eps
    β = max(β, 0.0)
    β = min(β, BETA_CLAMP_MAX_REF[])
    β * BETA_FORWARD_SCALE_REF[]
end

@inline function lambda_gamma_slope(s::Float64, t_raw::Float64, phys)::Float64
    a  = 1.0 + softplus(phys.a_raw)
    b  = softplus(phys.b_raw)
    τ  = 1e-3 + softplus(phys.tau_raw)
    t0 = softplus(phys.toff_raw)
    t  = max(t0 + τ*t_raw, TMIN)
    return (a-1.0)/t - b
end

# Build initial θ from data
function _init_theta()
    seeds = seed_phys_from_data(Xn, μ, σ, S_CENTERS)
    return ComponentVector((
        phys = (
            a_raw    = invsoftplus(seeds.a - 1.0),
            b_raw    = invsoftplus(seeds.b),
            tau_raw  = invsoftplus(seeds.τ - 1e-3),
            toff_raw = invsoftplus(seeds.t_off),
            beta_raw = invsoftplus(0.02 + 1e-3),
        ),
        nn = p_nn
    ))
end

# RHS f(u)
lap_eval(x::AbstractVector) = LAPL * x
function f_eval(u::AbstractVector, theta, s::Real, t_raw::Real, deg::Vector{Int}, gnn_model, st)
    mode = _normalize_mode(CURRENT_MODE[])
    phys = theta.phys
    λ = (mode == :nn_first) ? 0.0 : lambda_gamma_slope(s, t_raw, phys)
    β = (mode == :nn_first) ? 0.0 : beta_effective(phys)

    urelu = @. max(u, 0.0); m_u = mean(u)
    if β > BETA_TOL && mode != :nn_first
        lap = lap_eval(u); agg = deg .* u .- lap
        N = length(u); F = reshape(vcat(u', urelu', agg', fill(s, N)', fill(m_u, N)'), 5, N)
        corr = vec(gnn_apply(gnn_model, theta.nn, st, F))
        return @. λ*u - β*lap + corr
    else
        agg = deg .* u
        N = length(u); F = reshape(vcat(u', urelu', agg', fill(s, N)', fill(m_u, N)'), 5, N)
        corr = vec(gnn_apply(gnn_model, theta.nn, st, F))
        return @. λ*u + corr
    end
end

# Micro-diffusion & projection
function micro_diffuse(u::AbstractVector; α::Real=MICRO_ALPHA_REF[], steps::Int=MICRO_DIFF_STEPS_REF[])
    ρ = LAPL_RHO_REF[] > 0 ? LAPL_RHO_REF[] : 1.0
    τ = clamp(α / ρ, 1e-8, 1.0)
    v = u
    @inbounds for _ in 1:steps
        v = v .- τ .* (LAPL * v)
    end
    v
end

function postprocess_step(upred_norm::AbstractVector; μ::AbstractVector, σ::AbstractVector,
    target_E::Float64, apply_micro::Bool=true)

    v = denorm_global(upred_norm, μ, σ)
    vpos = @. max(v, 0.0)
    E_pred = sum(vpos)
    if E_pred ≤ ENERGY_EPS
        idx = argmax(v); v_new = zeros(length(v)); v_new[idx] = target_E
        out = (v_new .- μ) ./ σ
        out = apply_micro ? micro_diffuse(out) : out
        v2  = denorm_global(out, μ, σ); v2pos = @. max(v2, 0.0)
        E2  = sum(v2pos) + ENERGY_EPS
        out = ((v2 .* (target_E / E2)) .- μ) ./ σ
        return out, true
    end
    α = target_E / (E_pred + ENERGY_EPS)
    v_scaled = v .* α
    out = (v_scaled .- μ) ./ σ
    out = apply_micro ? micro_diffuse(out) : out
    v2  = denorm_global(out, μ, σ); v2pos = @. max(v2, 0.0)
    E2  = sum(v2pos) + ENERGY_EPS
    out = ((v2 .* (target_E / E2)) .- μ) ./ σ
    return out, true
end

function step_predict(u::AbstractVector, theta, s::Real, t_raw::Real, Δs::Real;
        target_E::Union{Nothing,Float64}=nothing,
        apply_projection::Bool=false,
        apply_microdiff::Bool=MICRO_DIFF_ON[])

    du_m  = Δs .* f_eval(u, theta, s, t_raw, deg, gnn, st)
    upred = u .+ du_m
    if apply_projection && target_E !== nothing
        upost, _ = postprocess_step(upred; μ=μ, σ=σ, target_E=target_E, apply_micro=apply_microdiff)
        return upost
    else
        return apply_microdiff ? micro_diffuse(upred) : upred
    end
end
