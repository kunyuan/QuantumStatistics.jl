using Distributed

@everywhere using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, Statistics, BenchmarkTools, InteractiveUtils, Parameters, DelimitedFiles

# claim all globals to be constant, otherwise, global variables could impact the efficiency
@everywhere const kF, m = 1.919, 0.5
@everywhere const β = 25.0 / kF^2

@everywhere const Nk,Nt = 32,32
@everywhere const extQ = [@SVector [q, 0.0, 0.0] for q in range(0.0, stop=3.0 * kF, length=Nk)]
@everywhere const extT = range(0.0, stop=β, length=Nt)

obs = readdlm("gamma.csv", ',', Float64, '\n')

obs = reshape(obs,(Nk,Nk,Nt,Nt))

for (idx, t) in enumerate(extT)
    t = t[1]
    @printf("%10.6f  %10.6f\n", t, obs[idx])
end


for (idx, q) in enumerate(extQ)
    q = q[1]
    @printf("%10.6f  %10.6f\n", q, obs[idx,idx,Int(Nt/2),Int(Nt/2)])
end


