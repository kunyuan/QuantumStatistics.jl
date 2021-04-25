using Distributed

const Ncpu = 4
const totalStep = 1e8
const Repeat = 8

addprocs(Ncpu)

@everywhere using QuantumStatistics, LinearAlgebra, Random, Printf, StaticArrays, Statistics, BenchmarkTools, InteractiveUtils, Parameters, DelimitedFiles

# claim all globals to be constant, otherwise, global variables could impact the efficiency
@everywhere const kF, m = 1.919, 0.5
@everywhere const β = 25.0 / kF^2

@everywhere const obs1, obs2 = zeros(Float64, 1), zeros(Float64, 1)

@everywhere function WRPA(τ,q)
    # temporarily using a toy model
    g=1.0

    factor=1.0
    q2=dot(q,q)
    if τ<-β
        τ=τ+2*β
    elseif τ<0
        τ=τ+β
        factor=1.0
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
    T, K = config.var[1], config.var[2]
        # In case the compiler is too stupid, it is a good idea to explicitly specify the type here
    q = K[1]
    t = T[1]

    k1 = @SVector[kF,0,0]
    k2 = @SVector[kF,0,0]
    t1 = T[2]
    t2 = T[3]

    ω1 = (dot(q+k1, q+k1) - kF^2) * β
    τ1 = (t1-t)/β
    g1 = Spectral.kernelFermiT(τ1, ω1)

    ω2 = (dot(q+k2, q+k2) - kF^2) * β
    τ2 = (β-t1-t)/β
    g2 = -Spectral.kernelFermiT(τ2, ω2)

    W1 = WRPA(t, q)
    W2 = WRPA(t-t1-t2,q)

    legendre=1.0
    factor = 1.0/(2π)^3
    return g1 * g2 * W1 * W2 * factor * legendre
end

@everywhere function measure(config)
    diag = config.curr
    factor = 1.0 / diag.reWeightFactor
    if diag.id == 1
        obs1[1] += factor
    elseif diag.id == 2
        weight = integrand(config)
        obs2[1] += weight / abs(weight) * factor
    else
        return
    end
end

@everywhere function MC(totalStep, pid)
    rng = MersenneTwister(pid)

    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    T = MonteCarlo.Tau(β, β / 2.0)
    Theta = MonteCarlo.Angle(0.5)
    diag1 = MonteCarlo.Diagram(1, 0, [0, 0])
    diag2 = MonteCarlo.Diagram(2, 1, [3, 1])

    config = MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K); pid=pid, rng=rng)

    # @code_warntype MonteCarlo.Configuration(totalStep, (diag1, diag2), (T, K, Ext); pid=pid, rng=rng)
    # @code_warntype MonteCarlo.Diagram(2, 1, [2, 1, 1], zeros(Float64, Ext.size))
    # @code_warntype MonteCarlo.increaseOrder(config, integrand)
    # @code_warntype integrand(config)
    # @code_warntype eval2(config)
    # @code_warntype measure(config)
    # @code_lowered MonteCarlo.changeVar(config, integrand)
    # exit()

    MonteCarlo.montecarlo(config, integrand, measure)

    return obs2 / sum(obs1)
end

function run(repeat, totalStep)
    if Ncpu > 1
        observable = pmap((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    else
        observable = map((x) -> MC(totalStep, rand(1:10000)), 1:repeat)
    end

    obs = mean(observable)
    obserr = std(observable) / sqrt(length(observable))

    @printf("%10.6f ± %10.6f\n",  obs[1], obserr[1])

end

# @btime run(1, 10)
run(Repeat, totalStep)
# @time run(Repeat, totalStep)
