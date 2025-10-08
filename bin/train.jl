#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
try Pkg.instantiate() catch; end

include(joinpath(@__DIR__, "..", "src", "PhysicsGuidedECAL.jl"))
using .PhysicsGuidedECAL

PhysicsGuidedECAL.main(ARGS)
