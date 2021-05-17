using Printf, LinearAlgebra, Distributed

const Ncpu = 16 # number of workers (CPU)
const totalStep = 1e9
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
        qgrid = Grid.boseK(kF, 6kF, 0.1kF, 256) 
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

    K, ExtT, ExtK = config.var[1], config.var[2], config.var[3]
    k = K[1]
    extTidx, extKidx = ExtT[1], ExtK[1]
    k0 = kgrid.grid[extKidx] # external momentum
    τ = dlr.τ[extTidx] # external τ
    kq = k + k0
    ω = (dot(kq, kq) - kF^2) / (2me)
    g = Spectral.kernelFermiT(τ, ω, β)
    v, dW = interactionDynamic(config, k, 0.0, τ)
    phase = 1.0 / (2π)^3
    return g * dW * phase
end

@everywhere function measure(config)
    factor = 1.0 / config.reweight[config.curr]
    ExtT, ExtK =  config.var[2], config.var[3]
    extTidx, extKidx = ExtT[1], ExtK[1]
    if config.curr == 1
        config.observable[1][1, 1] += factor
    elseif config.curr == 2
        weight = integrand(config)
        config.observable[2][extTidx, extKidx] += weight / abs(weight) * factor
    else
        return
    end
end

@everywhere normalize(config) = config.observable[2] / config.observable[1][1, 1]

function fock(extn)
    para = Para()
    println(para.τgrid.grid)
    Ksize = length(kgrid.grid)

    K = MonteCarlo.FermiK(dim, kF, 0.2 * kF, 10.0 * kF)
    ExtT = MonteCarlo.Discrete(1, dlr.size)
    ExtK = MonteCarlo.Discrete(1, Ksize) # external variable is specified

    dof = ([0, 0, 0], [1, 1, 1]) # degrees of freedom of the normalization diagram and the bubble
    obs = (zeros(Float64, (dlr.size, Ksize)), zeros(Float64, (dlr.size, Ksize))) # observable for the normalization diagram and the bubble

    avg, std = MonteCarlo.sample(totalStep, (K, ExtT, ExtK), dof, obs, integrand, measure, normalize; para=para, print=10)

    println("Tau: ")

    ki = findall(x -> x ≈ kF, kgrid.grid)[1]
    for (τi, τ) in enumerate(dlr.τ)
        @printf("%10.6f   %10.6f ± %10.6f\n", τ, avg[τi, ki], std[τi, ki])
    end

    println("Matsubara frequency: ")

    sigma = DLR.tau2matfreq(:fermi, avg, dlr, extn, axis=1)
    for (ni, n) in enumerate(extn)
        @printf("%10.6f   %10.6f    %10.6f\n", n, real(sigma[ni, ki]), imag(sigma[ni, ki]))
    end

    dS_dw = imag(sigma[1, ki] - sigma[2, ki]) / (2π / β)
    println("dΣ/diω=", dS_dw)
    println("Z=", 1 / (1 + dS_dw))
    # TODO: add errorbar estimation
    # println
end

if abspath(PROGRAM_FILE) == @__FILE__
    # using Gaston
    ngrid = [-1, 0, 1]
    fock(ngrid)
end