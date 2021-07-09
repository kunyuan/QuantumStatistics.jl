# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
using Roots, Polylogarithms
# using ProfileView

const Steps = 1e6

include("parameter.jl")

function chemical_potential(beta)
    f(β, μ) = real(polylog(3 / 2, -exp(β * μ))) + 4 / 3 / π^0.5 * (β)^(3 / 2)
    g(μ) = f(beta, μ)
    return find_zero(g, (-10000, 1))
end

@with_kw struct Para
    n::Int = 0 # external Matsubara frequency
    Qsize::Int = 16
    extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0, 3.0 * kF, Qsize)]
    μ::Float64 = chemical_potential(beta) * EF
end

function integrand(config)
    if config.curr != 1
        error("impossible")
    end
    para = config.para

    μ = para.μ
    T, K, Ext = config.var[1], config.var[2], config.var[3]
    k1, k2 = K[1], K[2]
    T0, T1, T2, T3 = 0.0, T[1], T[2], T[3]
    extidx = Ext[1]
    q = para.extQ[extidx] # external momentum
    k1q = k1 + q
    k2q = k2 + q
    dk = k1 - k2
    g1 = Spectral.kernelFermiT(T1 - T0, dot(k1q, k1q) / (2me) - μ, β)
    g2 = Spectral.kernelFermiT(T3 - T1, dot(k2q, k2q) / (2me) - μ, β)
    g3 = Spectral.kernelFermiT(T2 - T3, dot(k2, k2) / (2me) - μ, β)
    g4 = Spectral.kernelFermiT(T0 - T2, dot(k1, k1) / (2me) - μ, β)
    v = 8π / (dot(dk, dk) + mass2)
    phase = 1.0 / (2π)^3 / (2π)^3
    return g1 * g2 * g3 * g4 * v * spin * phase * cos(2π * para.n * T3)
end

function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    extidx = config.var[3][1]
    weight = integrand(config)
    obs[extidx] += weight / abs(weight) * factor
end

function run(steps)

    para = Para()
    @unpack extQ, Qsize = para 

    # println("μ=$μ, reduced β=$beta")

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MonteCarlo.Discrete(1, length(extQ)) # external variable is specified

    dof = [[3, 2, 1]] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64, Qsize) # observable for the normalization diagram and the bubble

    config = MonteCarlo.Configuration(steps, (T, K, Ext), dof, obs; para=para)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    if isnothing(avg) == false
        @unpack n, extQ = Para()

        for (idx, q) in enumerate(extQ)
            q = q[1]
            p, err = TwoPoint.LindhardΩnFiniteTemperature(dim, q, n, para.μ, kF, β, me, spin)
            @printf("%10.6f  %10.6f ± %10.6f  %10.6f ± %10.6f\n", q / kF, avg[idx], std[idx], p, err)
        end

    end
end

run(Steps)
# @time run(Steps)