# Modes & helpers
@inline softplus(x)    = log1p(exp(-abs(x))) + max(x, 0)
@inline invsoftplus(y) = log(exp(y) - 1)

function _normalize_mode(x)::Symbol
    x === true  && return :full
    x === false && return :nn_first
    x isa Symbol && return x
    @warn "Unknown mode '$x', defaulting to :full"
    return :full
end
function _normalize_target(x)::Symbol
    x === true  && return :conserve
    x === false && return :none
    x isa Symbol && return x
    @warn "Unknown energy target '$x', defaulting to :conserve"
    return :conserve
end
set_mode!(x)    = (CURRENT_MODE[]   = _normalize_mode(x))
set_energy!(x)  = (ENERGY_TARGET[]  = _normalize_target(x))
set_beta_scale!(r::Real) = (BETA_FORWARD_SCALE_REF[] = max(r, 0.0))
set_beta_clamp!(c::Real) = (BETA_CLAMP_MAX_REF[]     = max(c, 0.0))
enable_microdiff!(on::Bool, α::Real=MICRO_ALPHA_REF[]) = (MICRO_DIFF_ON[] = on; MICRO_ALPHA_REF[] = clamp(α, 1e-6, 0.99))

# Grid/Laplacian
function grid_neighbors(H::Int, W::Int)
    N = H*W
    to_id(i,j) = (i-1)*W + j
    nbrs = [Int[] for _ in 1:N]; deg = fill(0, N)
    for i in 1:H, j in 1:W
        u = to_id(i,j)
        if i>1; push!(nbrs[u], to_id(i-1,j)); end
        if i<H; push!(nbrs[u], to_id(i+1,j)); end
        if j>1; push!(nbrs[u], to_id(i,j-1)); end
        if j<W; push!(nbrs[u], to_id(i,j+1)); end
        deg[u] = length(nbrs[u])
    end
    return nbrs, deg
end

function build_laplacian_sparse(H::Int, W::Int, nbrs::Vector{Vector{Int}}, deg::Vector{Int})
    N = H * W
    rows = Int[]; cols = Int[]; vals = Float64[]
    nnz_est = N + sum(length(n) for n in nbrs)
    sizehint!(rows, nnz_est); sizehint!(cols, nnz_est); sizehint!(vals, nnz_est)
    for i in 1:N
        push!(rows, i); push!(cols, i); push!(vals, deg[i])
        for j in nbrs[i]
            push!(rows, i); push!(cols, j); push!(vals, -1.0)
        end
    end
    sparse(rows, cols, vals, N, N)
end

function spectral_radius_pow(A::SparseMatrixCSC{Float64,Int}; iters::Int=30)
    n = size(A,1)
    x = randn(n); x = x ./ (norm(x) + EPS)
    λ = 0.0
    for _ in 1:iters
        y = A*x
        ny = norm(y)
        ny ≤ EPS && break
        x = y ./ ny
        λ = dot(x, A*x) / (dot(x,x) + EPS)
    end
    abs(λ)
end

# Normalization & energy helpers
denorm_global(u_norm::AbstractVector, μ::AbstractVector, σ::AbstractVector) = u_norm .* σ .+ μ
function energy_from_normalized(u_norm::AbstractVector, μ::AbstractVector, σ::AbstractVector)
    v = denorm_global(u_norm, μ, σ)
    return sum(max.(v, 0.0))
end
energy_from_normalized(U::AbstractArray, μ::AbstractVector, σ::AbstractVector) =
    energy_from_normalized(vec(U), μ, σ)
