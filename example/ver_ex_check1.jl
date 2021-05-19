
# calculate (d) diagram in Kohn-Luttinger's paper as kernel of gap-equation at kF on dlr tau grids

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
# using ProfileView

const Steps = 1e7

include("parameter.jl")
include("../application/electron_gas/RPA.jl")

dlr = DLR.DLRGrid(:fermi, 10*EF,β, 1e-2)

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
    N = config.var[3]
    n1,n2 = dlr.n[N[1]],dlr.n[N[2]]
    if config.curr == 1
        T  ,K= config.var[1],config.var[2]
        q = K[1]
        t1 = T[1]

        k1 = kF * @SVector[1, 0, 0]
        k2 = kF * @SVector[1, 0, 0]

        ω1 = (dot(q-k1, q-k1) - kF^2) * β

        ω2 = (dot(q-k2, q-k2) - kF^2) * β

        τ3 = (t1)/β
        g3 = Spectral.kernelFermiT(τ3, ω1)

        τ6 = (-t1)/β
        g6 = Spectral.kernelFermiT(τ6, ω2)

        # W1 = WRPA(t, q-k1-k2)
        # W2 = WRPA(t-t1-t2,q)

        W1 = interaction(q-k1-k2, 0, 0/β)
        W2 = interaction(q, (t1)/β,t1/β )

        s_s = W1[1]*W2[1]*g3*g6*cos(π*(2*n1+1) * t1 /β) * cos(π*(2*n2+1) * t1/β)*β*β

        factor = 1.0/(2π)^3
        return (s_s) * factor
    elseif config.curr == 2
        T  ,K= config.var[1],config.var[2]
        q = K[1]
        t1, t2 = T[1], T[2]

        k1 = kF * @SVector[1, 0, 0]
        k2 = kF * @SVector[1, 0, 0]

        ω1 = (dot(q-k1, q-k1) - kF^2) * β

        ω2 = (dot(q-k2, q-k2) - kF^2) * β

        τ3 = (t1)/β
        g3 = Spectral.kernelFermiT(τ3, ω1)

        τ4 = (-t2)/β
        g4 = Spectral.kernelFermiT(τ4, ω2)

        τ5 = (t2)/β
        g5 = Spectral.kernelFermiT(τ5, ω1)

        τ6 = (-t1)/β
        g6 = Spectral.kernelFermiT(τ6, ω2)

        # W1 = WRPA(t, q-k1-k2)
        # W2 = WRPA(t-t1-t2,q)


        W10 = interaction(q-k1-k2,0,(t1-t2)/β)
        W20 = interaction(q,t2/β,t1/β)

        s_r = W10[1]*W20[2]*g3*g4*cos(π*(2*n1+1) * t1 /β) * cos(π*(2*n2+1) * t2/β)
        r_s = W10[2]*W20[1]*g5*g6*cos(π*(2*n1+1) * t1 /β) * cos(π*(2*n2+1) * t2/β)

        factor = 1.0/(2π)^3 
        return (s_r+r_s) * factor

    elseif config.curr == 3
        T  ,K= config.var[1],config.var[2]
        q = K[1]
        t, t1, t2 = T[1], T[2], T[3]

        k1 = kF * @SVector[1, 0, 0]
        k2 = kF * @SVector[1, 0, 0]

        ω1 = (dot(q-k1, q-k1) - kF^2) * β
        τ1 = (t1-t)/β
        g1 = Spectral.kernelFermiT(τ1, ω1)

        ω2 = (dot(q-k2, q-k2) - kF^2) * β
        τ2 = (β-t2-t)/β
        g2 = -Spectral.kernelFermiT(τ2, ω2)

        W1 = interaction(q-k1-k2, 0, t/β)
        W2 = interaction(q, (t+t2)/β,t1/β )

        r_r = W1[2]*W2[2]*g1*g2*cos(π*(2*n1+1) * t1 /β) * cos(π*(2*n2+1) * t2/β)/β/β

        factor = 1.0/(2π)^3 
        return (r_r) * factor

    else
        return 0.0
    end

end

function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    if config.curr!=1
        obs[config.var[3][1],config.var[3][2]] += weight / abs(weight) * factor
    end
end

function run(steps)

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    N = MonteCarlo.Discrete(1, length(dlr.n))

    dof = [[1,1,2],[2,1,2],[3,1,2]] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64,(length(dlr.n),length(dlr.n))) # observable forp the normalization diagram and the bubble
    println(size(obs))
    nei = [[2,4],[3,1],[2,4],[1,3]]

    config = MonteCarlo.Configuration(steps, (T,K,N ), dof, obs; neighbor = nei)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    # NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
    # println("NF=$NF")

    avg_tw = real(DLR.matfreq2tau(:fermi,avg,dlr,dlr.τ;axis=1,rtol=1e-12))
    avg_tt = real(DLR.matfreq2tau(:fermi,avg_tw,dlr,dlr.τ;axis=2,rtol=1e-12))

    avg_wt = real(DLR.tau2matfreq(:fermi,avg_tt,dlr,dlr.n;axis=1,rtol=1e-12))
    avg_ww = real(DLR.tau2matfreq(:fermi,avg_wt,dlr,dlr.n;axis=2,rtol=1e-12))
    println(size(avg_ww))

    if isnothing(avg) == false
        for i in 1:length(dlr.n)
            for j in 1:length(dlr.n)
                @printf("%d\t %d\t %10.6f ± %10.6f\t %10.6f\t %10.6f\n",
                        dlr.n[i],dlr.n[j], avg[i,j], std[i,j], avg_ww[i,j], (avg[i,j]-avg_ww[i,j])/std[i,j])
            end
        end
        println("tau-tau")
        for i in 1:length(dlr.n)
            for j in 1:length(dlr.n)
                @printf("%10.6f\t %10.6f\t %10.6f\t\n",
                        dlr.τ[i],dlr.τ[j], avg_tt[i,j])
            end
        end
    end
end

run(Steps)
# @time run(Steps)
