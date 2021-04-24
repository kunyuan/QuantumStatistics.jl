# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using Distributed

const Ncpu = 8 
const totalStep = 1e7
const Repeat = 8

addprocs(Ncpu)

@everywhere using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, Statistics, BenchmarkTools, InteractiveUtils, Parameters

# claim all globals to be constant, otherwise, global variables could impact the efficiency
@everywhere const kF, m, Qsize = 1.919, 0.5, 16
@everywhere const β = 25.0 / kF^2
@everywhere const n = 0 # external Matsubara frequency
@everywhere const extQ = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0, 3.0 * kF, Qsize)]
@everywhere const obs1, obs2 = zeros(Float64, Qsize), zeros(Float64, Qsize)

@everywhere function integrand(config)
    if config.curr.id == 1
        return 1.0
    elseif config.curr.id == 2
            return eval2(config)
        else
            return 0.0
    end
end

@everywhere function eval2(config)
    T, K, Ext = config.var[1], config.var[2], config.var[3]
        # In case the compiler is too stupid, it is a good idea to explicitly specify the type here
    k = K[1]
    Tin, Tout = T[1], T[2] 
    extidx = Ext[1]
    q = extQ[extidx] # external momentum
    kq = k + q
    τ = (Tout - Tin) / β
    ω1 = (dot(k, k) - kF^2) * β
    g1 = Spectral.kernelFermiT(τ, ω1)
    ω2 = (dot(kq, kq) - kF^2) * β
    g2 = Spectral.kernelFermiT(-τ, ω2)
    spin = 2
    phase = 1.0 / (2π)^3
    return g1 * g2 * spin * phase * cos(2π * n * τ)
    # return g1 * g2 * spin * phase
end

@everywhere function measure(config)
    diag = config.curr
    factor = 1.0 / diag.reWeightFactor
    extidx = config.var[3][1]
    if diag.id == 1
        obs1[extidx] += factor
    elseif diag.id == 2
        weight = integrand(config)
        obs2[extidx] += weight / abs(weight) * factor
    else
        return
    end
end

@everywhere function MC(totalStep, pid)
    rng = MersenneTwister(pid)

    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    T = MonteCarlo.Tau(β, β / 2.0)
    Ext = MonteCarlo.Discrete(1, length(extQ)) # external variable is specified
    diag1 = MonteCarlo.Diagram(1, 0, [1, 0, 1])
    diag2 = MonteCarlo.Diagram(2, 1, [2, 1, 1])

    config = MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, Ext); pid=pid, rng=rng)

    # @code_warntype MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, Ext); pid=pid, rng=rng)
    # @code_warntype MonteCarlo.Diagram(2, 1, [2, 1, 1], zeros(Float64, Ext.size))
    # @code_warntype MonteCarlo.increaseOrder(config, integrand)
    # @code_warntype integrand(config)
    # @code_warntype eval2(config)
    # @code_warntype measure(config)
    # @code_lowered MonteCarlo.changeVar(config, integrand)
    # exit()

    MonteCarlo.montecarlo(config, integrand, measure)

    return obs2 / sum(obs1) * Ext.size
end

function run(repeat, totalStep)
    if Ncpu > 1
        observable = pmap((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    else
        observable = map((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    end

    obs = mean(observable)
    obserr = std(observable) / sqrt(length(observable))

    for (idx, q) in enumerate(extQ)
        q = q[1]
        p, err = TwoPoint.LindhardΩnFiniteTemperature(3, q, n, kF, β, m, 2)
        @printf("%10.6f  %10.6f ± %10.6f  %10.6f ± %10.6f\n", q / kF, obs[idx], obserr[idx], p, err)
    end
end

# @btime run(1, 10)
run(Repeat, totalStep)
# @time run(Repeat, totalStep)