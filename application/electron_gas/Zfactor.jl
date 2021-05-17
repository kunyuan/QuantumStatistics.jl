using Printf, LinearAlgebra, Distributed

const Ncpu = 16 # number of workers (CPU)
const totalStep = 1e8
addprocs(Ncpu)

@everywhere using QuantumStatistics, Parameters, Random, LinearAlgebra
@everywhere include("parameter.jl")
@everywhere include("interaction.jl")
@everywhere include("RPA.jl")

@everywhere const kgrid = Grid.fermiK(kF, 3kF, 0.2kF, 16)  # external K grid for sigma
@everywhere const dlr = DLR.DLRGrid(:fermi, 10EF, β, 1e-10)

@everywhere struct Para{Q,T}
    dW0::Matrix{Float64}
    qgrid::Q
    τgrid::T # dedicated τgrid for dynamic interaction
    function Para()
        qgrid = Grid.boseK(kF, 6kF, 0.1kF, 512) 
        # τgrid = Grid.tau(β, EF / 5, 128) #for rs=4
        τgrid = Grid.tau(β, EF / 25, 128) # for rs=1
        # TODO: τgrid halflife works very strange

        vqinv = [(q^2 + mass2) / (4π * e0^2) for q in qgrid.grid]
        dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, me) # dynamic part of the effective interaction
        return new{typeof(qgrid),typeof(τgrid)}(dW0, qgrid, τgrid)
    end
end

@everywhere function integrand(config)
    if config.curr == 1
        return 1.0
    elseif config.curr == 2
        return eval2(config)
    else
        error("impossible!")
    end
end

@everywhere function eval2(config)
    para = config.para

    K, T = config.var[1], config.var[2]
    k, τ = K[1], T[1]
    k0 = [0.0, 0.0, kF] # external momentum
    kq = k + k0
    ω = (dot(kq, kq) - kF^2) / (2me)
    g = Spectral.kernelFermiT(τ, ω, β)
    v, dW = interactionDynamic(config, k, 0.0, τ)
    phase = 1.0 / (2π)^3
    return g * dW * phase
end

@everywhere function measure(config)
    factor = 1.0 / config.reweight[config.curr]
    τ = config.var[2][1]
    if config.curr == 1
        config.observable[1][1] += factor
    elseif config.curr == 2
        weight = integrand(config)
        config.observable[2][1] += weight * sin(-π / β * τ) / abs(weight) * factor
        config.observable[2][2] += weight * sin(π / β * τ) / abs(weight) * factor
    else
        return
    end
end

@everywhere normalize(config) = config.observable[2] / config.observable[1][1]

function fock(extn)
    para = Para()
    Ksize = length(kgrid.grid)

    K = MonteCarlo.FermiK(dim, kF, 0.2 * kF, 10.0 * kF)
    T = MonteCarlo.Tau(β, β / 2.0)

    dof = ([0, 0], [1, 1]) # degrees of freedom of the normalization diagram and the bubble
    obs = ([0.0, 0.0], [0.0, 0.0]) # observable for the normalization diagram and the bubble

    avg, std = MonteCarlo.sample(totalStep, (K, T), dof, obs, integrand, measure, normalize; para=para, print=10)


    @printf("%10.6f   %10.6f ± %10.6f\n", -1.0, avg[1], std[1])
    @printf("%10.6f   %10.6f ± %10.6f\n", 0.0, avg[2], std[2])

    dS_dw = (avg[1] - avg[2]) / (2π / β)
    error = (std[1] + std[2]) / (2π / β)
    println("dΣ/diω= $dS_dw ± $error") 
    Z = (1 / (1 + dS_dw))
    Zerror = error / Z^2
    println("Z=  $Z ± $Zerror")
    # TODO: add errorbar estimation
    # println
end

if abspath(PROGRAM_FILE) == @__FILE__
    # using Gaston
    ngrid = [-1, 0, 1]
    fock(ngrid)
end