module PhysicsGuidedECAL

using LinearAlgebra, Statistics, Random
using Optimization, OptimizationOptimJL
using Optimisers
using Lux, ComponentArrays, Zygote
using HDF5, StableRNGs, SparseArrays
using Logging, Printf, Dates
using Plots
using EzXML
using JSON
using JLD2
using SciMLBase

# Make subfiles visible inside the same module namespace
include("cli.jl")
include("constants.jl")
include("utils.jl")
include("data.jl")
include("model.jl")
include("training.jl")
include("lbfgs.jl")
include("evalplot.jl")

export main

"""
main(args)

End-to-end runner:
1) parse CLI & load data
2) Stage A (Adam)
3) Stage B (LBFGS)
4) Evaluate, plot, and save metrics/checkpoint
"""
function main(args::Vector{String})
    opt = _parse_cli(args)
    if get(opt, "help", false)
        _print_help()
        return
    end

    # Set smoke-mode if asked
    if get(opt, "smoke", false); SMOKE_MODE[] = true; end

    # Prepare dataset & graph
    prepare!(; h5=get(opt,"h5",H5), xml=get(opt,"xml",CC_XML), max_events=get(opt,"max",MAX_EVENTS))

    # Initialize params from data
    global theta = _init_theta()
    global theta0 = deepcopy(theta)

    # Stage A (Adam)
    info("=== Stage A (Adam) ===")
    train_stage_a!()

    # Stage B (LBFGS, physics on & frozen)
    info("=== Stage B (LBFGS) ===")
    train_stage_b!()

    # Evaluate & save
    info("=== Evaluation & Save ===")
    evaluate_and_save!()
end

end # module
