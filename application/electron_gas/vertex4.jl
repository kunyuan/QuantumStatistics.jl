# This example demonstrated how to calculate one-loop diagram of free electrons using the Monte Carlo module
# Observable is normalized: Γ₄*N_F where N_F is the free electron density of states

using Distributed

const Ncpu = 8 # number of workers (CPU)
const totalStep = 1e7 # MC steps of each worker
const Repeat = Ncpu # total number of MC jobs
addprocs(Ncpu) 

@everywhere using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, Statistics, BenchmarkTools, InteractiveUtils, Parameters
@everywhere include("RPA.jl")

# claim all globals to be constant, otherwise, global variables could impact the efficiency
########################### parameters ##################################
@everywhere const kF, m, e, spin, AngSize = 1.919, 0.5, sqrt(2), 2, 32
@everywhere const mass2 = 0.01
@everywhere const β, EF = 100.0 / (kF^2 / 2m), kF^2 / (2m)
@everywhere const n = 0 # external Matsubara frequency
@everywhere const IsF = false # calculate quasiparticle interaction F or not
@everywhere const extAngle = collect(LinRange(0.0, π, AngSize)) # external angle grid

########################## variables for MC integration ##################
@everywhere const Weight = SVector{2,Float64}
@everywhere const Base.abs(w::Weight) = abs(w[1]) + abs(w[2]) # define abs(Weight)
@everywhere const obs1, obs2 = [0.0, ], zeros(Weight, AngSize)
@everywhere const KInL = [kF, 0.0, 0.0] # incoming momentum of the left particle
@everywhere const Qd = [0.0, 0.0, 0.0] # transfer momentum is zero in the forward scattering channel

################ construct RPA interaction ################################
@everywhere const qgrid = Grid.boseK(kF, 6kF, 0.2kF, 256) 
@everywhere const τgrid = Grid.tau(β, EF / 20, 128)
@everywhere const vqinv = [(q^2 + mass2) / (4π * e^2) for q in qgrid.grid]
@everywhere const dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, 2, m) # dynamic part of the effective interaction

# println(qgrid.grid)
# println(τgrid.grid)

@everywhere function interaction(qd, qe, τIn, τOut)
    dτ = abs(τOut - τIn)

    kDiQ = sqrt(dot(qd, qd))
    vd = 4π * e^2 / (kDiQ^2 + mass2)
    if kDiQ <= qgrid.grid[1]
        wd = vd * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-6, dτ) # the current interpolation vanishes at q=0, which needs to be corrected!
    else
        wd = vd * Grid.linear2D(dW0, qgrid, τgrid, kDiQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end

    kExQ = sqrt(dot(qe, qe))
    ve = 4π * e^2 / (kExQ^2 + mass2)
    if kExQ <= qgrid.grid[1]
        we = ve * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-6, dτ) # dynamic interaction, don't forget the singular factor vq
    else
        we = ve * Grid.linear2D(dW0, qgrid, τgrid, kExQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end

    return -vd / β, ve / β, -wd, we
end

@everywhere function phase(tInL, tOutL, tInR, tOutR)
    # return 1.0;
    if (IsF)
        return cos(π * ((tInL + tOutL) - (tInR + tOutR)));
    else
        return cos(π * ((tInL - tOutL) + (tInR - tOutR)))
    end
end

@everywhere function integrand(config)
    if config.curr.id == 1
        return Weight(1.0, 0.0) # return a weight!
    elseif config.curr.id == 2
        return eval2(config)
    else
        error("Not implemented!")
    end
end

@everywhere function eval2(config)
    T, K, Ang = config.var[1], config.var[2], config.var[3]
    k1, k2 = K[1], K[1] - Qd
    t1, t2 = T[1], T[2] # t1, t2 both have two tau variables
    θ = extAngle[Ang[1]] # angle of the external momentum on the right
    KInR = [kF * cos(θ), kF * sin(θ), 0.0]

    vld, vle, wld, wle = interaction(Qd, KInL - k1, t1[1], t1[2])
    vrd, vre, wrd, wre = interaction(Qd, KInR - k2, t2[1], t2[2])

    ϵ1, ϵ2 = (dot(k1, k1) - kF^2) / (2m), (dot(k2, k2) - kF^2) / (2m) 
    wd, we = 0.0, 0.0
    # possible green's functions on the top
    gt1 = Spectral.kernelFermiT(t2[1] - t1[1], ϵ1, β)


    # gt2 = Spectral.kernelFermiT(t1[1] - t2[1], ϵ2, β)
    # wd += 1.0 / β * 1.0 / β * gt1 * gt2 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])

    # gt3 = Spectral.kernelFermiT(t1[1] - t2[2], ϵ2, β)
    # G = gt1 * gt3 / (2π)^3 * phase(t1[1], t1[1], t2[2], t2[1])
    # wd += G * (vld * wre)

    # wd += spin * (vld + wld) * (vrd + wrd) * gt1 * gt2 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])
    # println(vld, ", ", wld, "; ", vrd, ", ", wld)

    ############## Diagram v x v ######################
    """
      KInL                      KInR
       |                         | 
  t1.L ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   ve  |          |  ve  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.L        t2.L     ↑ t2.L
       |                         | 
      KInL                      KInR
"""
    gd1 = Spectral.kernelFermiT(t1[1] - t2[1], ϵ2, β)
    G = gt1 * gd1 / (2π)^3 * phase(t1[1], t1[1], t2[1], t2[1])
    we += G * (vle * vre)
    ##################################################

    ############## Diagram w x v ######################
    """
      KInL                      KInR
       |                         | 
  t1.R ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   we  |          |  ve  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.R        t2.L     ↑ t2.L
       |                         | 
      KInL                      KInR
    """
    gd2 = Spectral.kernelFermiT(t1[2] - t2[1], ϵ2, β)
    G = gt1 * gd2 / (2π)^3 * phase(t1[1], t1[2], t2[1], t2[1])
    we += G * (wle * vre) 
    ##################################################

    ############## Diagram v x w ######################
    """
      KInL                      KInR
       |                         | 
  t1.L ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   ve  |          |  we  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.L        t2.R     ↑ t2.R
       |                         | 
      KInL                      KInR
    """
    gd3 = Spectral.kernelFermiT(t1[1] - t2[2], ϵ2, β)
    G = gt1 * gd3 / (2π)^3 * phase(t1[1], t1[1], t2[2], t2[1])
    we += G * (vle * wre)
    ##################################################

    ############## Diagram w x w ######################
    """
      KInL                      KInR
       |                         | 
  t1.R ↑     t1.L       t2.L     ↑ t2.L
       |-------------->----------|
       |       |    k1    |      |
       |   we  |          |  we  |
       |       |    k2    |      |
       |--------------<----------|
  t1.L ↑    t1.R        t2.R     ↑ t2.R
       |                         | 
      KInL                      KInR
"""
    gd4 = Spectral.kernelFermiT(t1[2] - t2[2], ϵ2, β)
    G = gt1 * gd4 / (2π)^3 * phase(t1[1], t1[2], t2[2], t2[1])
    we += G * (wle * wre)
    ##################################################

    return Weight(wd, we)
end

@everywhere function measure(config)
    diag = config.curr
    factor = 1.0 / config.absWeight / diag.reWeightFactor
    if diag.id == 1
        obs1[1] += factor
    elseif diag.id == 2
        weight = integrand(config)
        angidx = config.var[3][1]
        # obs2[angidx] += weight / abs(weight) * factor
        obs2[angidx] += weight * factor
    else
        error("Not implemented!")
    end
end

@everywhere function MC(totalStep, pid)
    rng = MersenneTwister(pid)

    T = MonteCarlo.TauPair(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MonteCarlo.Discrete(1, length(extAngle)) # external variable is specified
    diag1 = MonteCarlo.Diagram(1, 0, [1, 0, 1]) # id, order, [T num, K num, Ext num]
    diag2 = MonteCarlo.Diagram(2, 1, [2, 1, 1]) # id, order, [T num, K num, Ext num]

    config = MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, Ext); pid=pid, rng=rng)
    MonteCarlo.montecarlo(config, integrand, measure)

    return obs2 / obs1[1] * β * AngSize
end

function run(repeat, totalStep)
    if Ncpu > 1
        observable = pmap((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    else
        observable = map((x) -> MC(totalStep, 1), 1:repeat)
    end

    diobservable = []
    exobservable = []
    for obs in observable
        diobs = [w[1] for w in obs] # direct vertex4
        push!(diobservable, diobs)
        exobs = [w[2] for w in obs] # exchange vertex4
        push!(exobservable, exobs)
    end

    NF = TwoPoint.LindhardΩnFiniteTemperature(3, 0.0, 0, kF, β, m, spin)[1]
    println("NF = $NF")

    diobs = mean(diobservable) * NF
    diobserr = std(diobservable) / sqrt(length(diobservable)) * NF

    exobs = mean(exobservable) * NF
    exobserr = std(exobservable) / sqrt(length(exobservable)) * NF


    for (idx, angle) in enumerate(extAngle)
        @printf("%10.6f   %10.6f ± %10.6f  %10.6f ± %10.6f\n", angle, diobs[idx], diobserr[idx], exobs[idx], exobserr[idx])
    end
end

# @btime run(1, 10)
run(Repeat, totalStep)
# @time run(Repeat, totalStep)