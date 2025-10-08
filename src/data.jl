# Read XML (for flat showers)
function read_cc_xml(xmlpath::AbstractString)
    @assert !isempty(xmlpath) && isfile(xmlpath) "xml file is needed when showers are flat, pass --xml path"
    doc = EzXML.read(xmlpath); root = EzXML.root(doc)
    findint(paths...) = begin
        for p in paths
            node = EzXML.findfirst(p, root)
            if node !== nothing
                v = try parse(Int, strip(string(EzXML.text(node)))) catch; nothing end
                v !== nothing && return v
            end
        end
        nothing
    end
    findfloats(paths...) = begin
        for p in paths
            node = EzXML.findfirst(p, root)
            if node !== nothing
                txt = strip(string(EzXML.text(node)))
                vals = split(replace(txt, ',' => ' '))
                arr = Float64[]
                for s in vals
                    isempty(s) && continue
                    push!(arr, tryparse(Float64, s) === nothing ? parse(Float64, s) : tryparse(Float64, s))
                end
                !isempty(arr) && return arr
            end
        end
        Float64[]
    end
    nr = findint("//n_r","//nr","//nrad","//nRad","//n_rad")
    nα = findint("//n_alpha","//nphi","//nAlpha","//n_alpha")
    nz = findint("//n_z","//nz","//nZ","//n_z")
    @assert nr !== nothing && nα !== nothing && nz !== nothing "could not read grid sizes from xml"
    z_edges = findfloats("//z_edges","//zedges","//zEdges")
    return nr, nα, nz, z_edges
end

# Fallback HDF5 key
function _read_key_fallbacks(f::HDF5.File, key::String)
    if haskey(f, key); return Array(f[key]), key; end
    alt = (key == "/incident_energies") ? "/incident energies" :
          (key == "/incident energies") ? "/incident_energies" : ""
    if alt != "" && haskey(f, alt)
        @warn "dataset key '$key' not found; using '$alt'"
        return Array(f[alt]), alt
    end
    error("Key '$key' not found; available top-level keys: $(collect(keys(f)))")
end

# Loader (returns E×Z×R×A)
function load_calo_data(h5file::AbstractString; xml::AbstractString="")
    @assert isfile(h5file) "hdf5 not found: $h5file"
    h5open(h5file, "r") do f
        inc, used_ie = _read_key_fallbacks(f, "/incident_energies")
        @assert haskey(f, "/showers") || haskey(f, "showers") "'showers' dataset not found in $h5file"
        showers = haskey(f,"/showers") ? Array(f["/showers"]) : Array(f["showers"])
        if ndims(showers) == 2
            nr, nα, nz, z_edges = read_cc_xml(xml)
            E, F = size(showers)
            @assert F == nr*nα*nz "flat showers length does not match xml"
            Y4 = Array{Float64}(undef, E, nz, nr, nα)
            for e in 1:E
                v = reshape(@view(showers[e, :]), nr, nα, nz)
                @views Y4[e, :, :, :] = permutedims(v, (3,1,2))
            end
            return Y4, used_ie, (incident_energies=inc, z_edges=z_edges)
        elseif ndims(showers) == 4
            if !isempty(xml)
                nr, nα, nz, z_edges = read_cc_xml(xml)
                dims = size(showers)
                if dims[2] == nz && dims[3] == nr && dims[4] == nα
                    return Array{Float64}(showers), used_ie, (incident_energies=inc, z_edges=z_edges)
                elseif dims[2] == nz && dims[3] == nα && dims[4] == nr
                    return permutedims(showers, (1,2,4,3)), used_ie, (incident_energies=inc, z_edges=z_edges)
                elseif dims[2] == nr && dims[3] == nα && dims[4] == nz
                    return permutedims(showers, (1,4,2,3)), used_ie, (incident_energies=inc, z_edges=z_edges)
                else
                    error("4d showers shape does not match xml")
                end
            else
                return Array{Float64}(showers), used_ie, (incident_energies=inc, z_edges=Float64[])
            end
        else
            error("showers must be 2d or 4d")
        end
    end
end

# Dataset + normalization
function build_dataset(Y4::Array{Float64,4}; train_frac=0.8)
    E,L,H,W = size(Y4); N = H*W
    X  = [zeros(L,N) for _ in 1:E]
    for e in 1:E, l in 1:L
        X[e][l, :] = vec(Y4[e,l,:,:])
    end
    idx = collect(1:E); Random.shuffle!(StableRNGs.StableRNG(11), idx)
    ntr = (E > 1) ? clamp(round(Int, train_frac*E), 1, E-1) : 1
    tr, va = (E > 1) ? (idx[1:ntr], idx[(ntr+1):end]) : (idx, Int[])
    μ = vec(mean(reduce(vcat, X[tr]); dims=1))
    σ = vec(std(reduce(vcat, X[tr]);  dims=1)); σ = max.(σ, 1e-8)
    Xn = [(X[e] .- μ') ./ σ' for e in 1:E]
    return X, Xn, tr, va, μ, σ, H, W
end

# Prepare everything and fill module globals
function prepare!(; h5::AbstractString, xml::AbstractString="", max_events=MAX_EVENTS)
    _ff = isdefined(Main, :find_file) ? Main.find_file : (x->x)
    h5_path  = _ff(h5); xml_path = isempty(xml) ? "" : _ff(xml)
    @info "Paths" h5=h5_path xml=(isempty(xml_path) ? "(none)" : xml_path)
    @assert isfile(h5_path) "HDF5 not found: $(h5_path)"

    # Load and coerce to E×Z×R×A (more permissive loader lives in the monolithic reference,
    # here we use the strict path above; adjust as needed)
    events4d, _, meta = load_calo_data(h5_path; xml=xml_path)
    if max_events !== nothing
        events4d = events4d[1:min(size(events4d,1), max(1, Int(max_events))), :, :, :]
    end
    @info "loaded events" E=size(events4d,1) L=size(events4d,2) R=size(events4d,3) A=size(events4d,4)

    global X, Xn, train_idx, val_idx, μ, σ, H, W
    X, Xn, train_idx, val_idx, μ, σ, H, W = build_dataset(events4d)

    # Depth grid
    Lz = size(Xn[1], 1)
    z_edges = get(meta, :z_edges, Float64[])
    if !isempty(z_edges) && length(z_edges) == (Lz+1)
        t_edges_raw   = copy(z_edges)
        t_centers_raw = [0.5*(t_edges_raw[i]+t_edges_raw[i+1]) for i in 1:Lz]
        s_edges       = t_edges_raw ./ max(t_edges_raw[end], 1e-12)
    else
        t_edges_raw   = collect(range(0.0, length=Lz+1, stop=Lz*1.0))
        t_centers_raw = [i-0.5 for i in 1:Lz]
        s_edges       = t_edges_raw ./ max(t_edges_raw[end], 1e-12)
    end
    global T_CENTERS_RAW = t_centers_raw
    global S_CENTERS     = [0.5*(s_edges[i]+s_edges[i+1]) for i in 1:length(s_edges)-1]
    global TS_CACHE      = [S_CENTERS for _ in 1:length(Xn)]

    # Graph
    global nbrs, deg = grid_neighbors(H, W)
    global LAPL      = build_laplacian_sparse(H, W, nbrs, deg)
    global NPIX      = H*W
    if LAPL_RHO_REF[] <= 0
        try
            LAPL_RHO_REF[] = max(spectral_radius_pow(LAPL; iters=35), 2.0*maximum(deg))
        catch
            LAPL_RHO_REF[] = max(2.0*maximum(deg), 1.0)
        end
    end
    @info "spectral radius" rho=LAPL_RHO_REF[]

    if PIX_SUBSAMPLE_REF[] <= 0
        PIX_SUBSAMPLE_REF[] = min(2048, NPIX)
        if SMOKE_MODE[]; PIX_SUBSAMPLE_REF[] = min(512, NPIX); end
    end

    # Build GNN
    global gnn, p_nn, st = make_gnn(RNG)
    p_nn.layer_4.weight .= 0.0
    p_nn.layer_4.bias   .= 0.0

    # Eval subset
    function eval_events(train_idx::Vector{Int}, val_idx::Vector{Int})
        if !isempty(val_idx)       ; return val_idx
        elseif !isempty(train_idx) ; return train_idx
        else                       ; return collect(1:length(X))
        end
    end
    global EVAL_IDX_CACHED = begin
        ev = eval_events(train_idx, val_idx)
        length(ev) > EVAL_EVENTS_MAX ? ev[1:EVAL_EVENTS_MAX] : ev
    end

    # Batch size & buffers
    function _batch_indices()
        total = isempty(train_idx) ? length(X) : length(train_idx)
        bsz   = max(1, min(FAST_BATCH_MULT_REF[] * BATCH_BASE_REF[], total))
        return bsz
    end
    global BSZ     = _batch_indices()
    global idx_buf = Vector{Int}(undef, BSZ)
    global ALL_IF_NO_SPLIT = isempty(train_idx)

    return nothing
end
