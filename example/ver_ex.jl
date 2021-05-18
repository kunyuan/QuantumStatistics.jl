
# calculate (d) diagram in Kohn-Luttinger's paper as kernel of gap-equation at kF on dlr tau grids

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
# using ProfileView

const Steps = 1e7

include("parameter.jl")
include("../application/electron_gas/RPA.jl")


const qgrid = Grid.boseKUL(kF, 6kF, 0.000001*sqrt(me^2/β/kF^2), 15,4) 
const τgrid = Grid.tauUL(β, 0.0001, 11,4)
const vqinv = [(q^2 + mass2) / (4π * e0^2) for q in qgrid.grid]
println(qgrid.grid)
println(τgrid.grid)

const dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, me) # dynamic part of the effective interaction

function interaction(q, τIn, τOut)
    dτ = abs(τOut - τIn)

    kQ = sqrt(dot(q, q))
    v = 4π * e0^2 / (kQ^2 + mass2)
    if kQ <= qgrid.grid[1]
        w = v * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-14, dτ) # the current interpolation vanishes at q=0, which needs to be corrected!
    else
        w = v * Grid.linear2D(dW0, qgrid, τgrid, kQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end

    return -v / β, -w
end

function WRPA(τ,q)
    return interaction(q,0,τ)[2]
end

# function WRPA(τ,q)
#     # temporarily using a toy model
#     g=1.0

#     factor=1.0
#     q2=dot(q,q)
#     if τ<-β
#         τ=τ+2*β
#     elseif τ<0
#         τ=τ+β
#         factor=1.0
#     end
#     sq2g=sqrt(q2+g)
#     return -factor*4*π/(q2+g)/sq2g*(exp(-sq2g*τ)+exp(-sq2g*(β-τ)))/(1-exp(-sq2g*β))
# end

function integrand(config)
    if config.curr != 1
        error("impossible")
    end

    T, K, Theta = config.var[1], config.var[2], config.var[3]
    q = K[1]
    t, t1, t2 = T[1], T[2], T[3]
    θ = Theta[1]

    k1 = kF * @SVector[1, 0, 0]
    k2 = kF * @SVector[cos(θ), sin(θ), 0]

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

    return g1 * g2 * W1 * W2 * factor * legendre * cos(π * t1 /β) * cos(π * t2/β)
end

function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    obs[1] += weight / abs(weight) * factor
end

function run(steps)

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Angle = MonteCarlo.Angle()

    dof = [[3, 1, 1],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64,1) # observable forp the normalization diagram and the bubble

    config = MonteCarlo.Configuration(steps, (T, K, Angle), dof, obs)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    if isnothing(avg) == false
            @printf("%10.6f ± %10.6f\n", avg[1], std[1])
    end
end

run(Steps)
# @time run(Steps)
