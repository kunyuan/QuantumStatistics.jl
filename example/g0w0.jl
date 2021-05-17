# This example demonstrated how to calculate one-loop diagram of free electrons using the Monte Carlo module
# Observable is normalized: Γ₄*N_F where N_F is the free electron density of states

using Distributed
using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, BenchmarkTools, InteractiveUtils, Parameters

const Ncpu = 4 # number of workers (CPU)
const totalStep = 1e8 # MC steps of each worker

addprocs(Ncpu) 

@everywhere using QuantumStatistics, Parameters, StaticArrays, Random, LinearAlgebra
@everywhere include("../application/electron_gas/RPA.jl")

# parameters
@everywhere @with_kw struct Para
    kF::Float64 = (9π/4.0)^(1.0/3)
    m::Float64 = 0.5
    β::Float64 = 1000.0 / kF^2 *2m
    spin::Int = 2
    n::Int = 0 # external Matsubara frequency
end

################ construct RPA interaction ################################
@everywhere const para = Para()
@everywhere const e2tors = 2.0/(9π/4.0)^(1.0/3)
@everywhere const kF, m, e, spin, AngSize = para.kF, para.m, sqrt(1.0*e2tors), para.spin, 32
@everywhere const mass2 = 0.0#0.000000001
@everywhere const β, EF = para.β, kF^2 / (2m)

@everywhere const qgrid = Grid.boseKUL(kF, 6kF, 0.000001*sqrt(m^2/β/kF^2), 15,4) 
@everywhere const τgrid = Grid.tauUL(β, 0.0001, 11,4)
@everywhere const vqinv = [(q^2 + mass2) / (4π * e^2) for q in qgrid.grid]
println(qgrid.grid)
println(τgrid.grid)

@everywhere const dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, m) # dynamic part of the effective interaction


@everywhere function interaction(q, τIn, τOut)
    dτ = abs(τOut - τIn)

    kQ = sqrt(dot(q, q))
    v = 4π * e^2 / (kQ^2 + mass2)
    if kQ <= qgrid.grid[1]
        w = v * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-14, dτ) # the current interpolation vanishes at q=0, which needs to be corrected!
    else
        w = v * Grid.linear2D(dW0, qgrid, τgrid, kQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end

    return -v / β, -w
end

@everywhere function integrand(config)
    if config.curr == 1
        return 1.0 # return a weight!
    elseif config.curr == 2
        return eval2(config)
    else
        error("Not implemented!")
    end
end

@everywhere function eval2(config)
    para = config.para
    β, kF, m, spin = para.β, para.kF, para.m, para.spin

    T, K = config.var[1], config.var[2]
    k1= K[1]
    t1, t2 = T[1], T[2] 
    KIN = @SVector[kF,0,0]

    v,w = interaction(k1, t1, t2)

    kq=KIN+k1
    τ = (t2-t1) / β

    ϵ1 = (dot(kq, kq) - kF^2) / (2m)
    # possible green's functions on the top
    g1 = Spectral.kernelFermiT(τ, ϵ1)
    g0 = Spectral.kernelFermiT(t1 - t1, ϵ1, β)

    phase = 1.0 / (2π)^3
    return spin * g1 * w * phase * (-τ * β) * sin(π * (2*para.n+1) * abs(τ)) #+ g0 * v * phase 
end

@everywhere function measure(config)
    obs = config.observable
    curr = config.curr
    factor = 1.0 / config.reweight[curr]
    if curr == 1
        obs[1][1] += factor
    elseif curr == 2
        weight = integrand(config)
        obs[2][1] += weight / abs(weight) * factor
    else
        return
    end
end

@everywhere normalize(config) = config.observable[2] / sum(config.observable[1])

function run(totalStep)

    para = Para()
    @unpack kF, β = para 

    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    T = MonteCarlo.Tau(β, β / 2.0)

    dof = ([1, 0], [2, 1]) # degrees of freedom of the normalization diagram and the bubble

    obs = (zeros(Float64, 1), zeros(Float64, 1)) # observable for the normalization diagram and the bubble

    avg, std = MonteCarlo.sample(totalStep, (T, K), dof, obs, integrand, measure, normalize; para=para, print=10)


    @unpack kF, β, m, n = Para()

    @printf("%10.6f ± %10.6f\n", avg[1], std[1])
    @printf("%10.6f ± %10.6f\n", 1.0/(1.0 - avg[1]),std[1]/(1.0-avg[1])^2)
end

run(totalStep)
# @time run(totalStep)
