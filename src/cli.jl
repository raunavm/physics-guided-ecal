# CLI parsing + tiny help

function _parse_cli(args::Vector{String})
    opt = Dict{String,Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help"
            opt["help"] = true; i += 1
        elseif a == "--h5" && i < length(args)
            opt["h5"] = args[i+1]; i += 2
        elseif a == "--xml" && i < length(args)
            opt["xml"] = args[i+1]; i += 2
        elseif a == "--max-events" && i < length(args)
            opt["max"] = try parse(Int, args[i+1]) catch; error("--max-events needs integer") end; i += 2
        elseif a == "--smoke"
            opt["smoke"] = true; i += 1
        else
            @warn "Unknown CLI arg ignored: $a"; i += 1
        end
    end
    return opt
end

function _print_help()
    println("""
Usage:
  julia --project bin/train.jl --h5 data/dataset_2_1.hdf5 [--xml data/grid.xml] [--max-events N] [--smoke]

""")
end

# Try current and script folder
find_file(p::AbstractString) = begin
    if isempty(p); return p; end
    if isabspath(p) && (isfile(p) || isdir(p)); return p; end
    for base in (pwd(), @__DIR__)
        candidate = joinpath(base, p)
        if isfile(candidate) || isdir(candidate)
            return candidate
        end
    end
    return p
end
