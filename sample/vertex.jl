using Distributed

const Ncpu = 4
const totalStep = 1e9
const Repeat = 9

addprocs(Ncpu)

@everywhere using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, Statistics, BenchmarkTools, InteractiveUtils, Parameters, DelimitedFiles

# claim all globals to be constant, otherwise, global variables could impact the efficiency
@everywhere const kF, m = 1.919, 0.5
@everywhere const β = 25.0 / kF^2

@everywhere const Nk,Nt = 32,32
@everywhere const extQ = [@SVector [q, 0.0, 0.0] for q in range(0.0, stop=3.0 * kF, length=Nk)]
@everywhere const extT = range(0.0, stop=β, length=Nt)

@everywhere const obs1, obs2 = zeros(Float64, (Nk,Nk,Nt,Nt)), zeros(Float64, (Nk,Nk,Nt,Nt))

@everywhere function WRPA(τ,q)
    # temporarily using a toy model
    g=1.0

    factor=1.0
    q2=dot(q,q)
    if τ<-β
        τ=τ+2*β
    elseif τ<0
        τ=τ+β
        factor=-1.0
    end
    sq2g=sqrt(q2+g)
    return -factor*4*π/(q2+g)/sq2g*(exp(-sq2g*τ)+exp(-sq2g*(β-τ)))/(1-exp(-sq2g*β))
end


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
    T, K, ExtQ, ExtT, Theta = config.var[1], config.var[2], config.var[3], config.var[4], config.var[5]
        # In case the compiler is too stupid, it is a good idea to explicitly specify the type here
    q = K[1]
    t = T[1]
    θ = Theta[1]

    k1 = extQ[ExtQ[1]] # external momentum
    k2 = extQ[ExtQ[2]] # external momentum
    t1 = extT[ExtT[3]] # external momentum
    t2 = extT[ExtT[4]] # external momentum

    k2 = norm(k2) * @SVector[cos(θ), sin(θ), 0]

    ω1 = (dot(q-k1, q-k1) - kF^2) * β
    τ1 = (t1-t)/β
    g1 = Spectral.kernelFermiT(τ1, ω1)

    ω2 = (dot(q-k2, q-k2) - kF^2) * β
    τ2 = (β-t1-t)/β
    g2 = Spectral.kernelFermiT(τ2, ω2)

    W1 = WRPA(t, q-k1-k2)
    W2 = WRPA(t-t1-t2,q)

    legendre=1.0
    factor = 1.0
    return g1 * g2 * W1 * W2 * factor * legendre
end

@everywhere function measure(config)
    diag = config.curr
    factor = 1.0 / diag.reWeightFactor
    extqidx = [config.var[3][1],config.var[3][2]]
    exttidx = [config.var[4][1],config.var[4][2]]
    extidx = CartesianIndex(Tuple(x for x in vcat(extqidx, exttidx)))
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
    Theta = MonteCarlo.Angle(0.5)
    ExtQ = MonteCarlo.Discrete(1, length(extQ)) # external variable is specified
    ExtT = MonteCarlo.Discrete(1, length(extT)) # external variable is specified
    diag1 = MonteCarlo.Diagram(1, 0, [0, 0, 2, 2, 0])
    diag2 = MonteCarlo.Diagram(2, 1, [1, 1, 2, 2, 1])

    config = MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, ExtQ, ExtT, Theta); pid=pid, rng=rng)

    # @code_warntype MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, Ext); pid=pid, rng=rng)
    # @code_warntype MonteCarlo.Diagram(2, 1, [2, 1, 1], zeros(Float64, Ext.size))
    # @code_warntype MonteCarlo.increaseOrder(config, integrand)
    # @code_warntype integrand(config)
    # @code_warntype eval2(config)
    # @code_warntype measure(config)
    # @code_lowered MonteCarlo.changeVar(config, integrand)
    # exit()

    MonteCarlo.montecarlo(config, integrand, measure)

    return obs2 / sum(obs1) * ExtQ.size^2 * ExtT.size ^2
end

function run(repeat, totalStep)
    if Ncpu > 1
        observable = pmap((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    else
        observable = map((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    end

    obs = mean(observable)
    obserr = std(observable) / sqrt(length(observable))

    for (idx, t) in enumerate(extT)
        t = t[1]
        @printf("%10.6f  %10.6f ± %10.6f\n", t, obs[idx], obserr[idx])
    end

    writedlm("gamma.csv",obs,",")


end

# @btime run(1, 10)
run(Repeat, totalStep)
# @time run(Repeat, totalStep)
