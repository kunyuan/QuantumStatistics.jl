using QuantumStatistics, Test, StaticArrays, LinearAlgebra, Printf, Random, Statistics
# import Test: @test, @testset

if isempty(ARGS)
    include("montecarlo.jl")
    include("grid.jl")
    include("correlator.jl")
    include("math.jl")
else
    include(ARGS[1])
end
