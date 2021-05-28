
# calculate (d) diagram in Kohn-Luttinger's paper as kernel of gap-equation at kF on dlr tau grids

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
# using ProfileView
using PyCall
const Steps = 4e7

include("parameter.jl")
include("../application/electron_gas/RPA.jl")

const ℓ=0

special = pyimport("scipy.special")
coeff = SVector{ℓ+1,Float64}(special.legendre(ℓ).c)
expn = zeros(ℓ+1)
for i=1:ℓ+1
    expn[i]=ℓ+1-i
end
expn = SVector{ℓ+1,Float64}(expn)

function legendre(x)
    return dot((x .^ expn) ,coeff)
end

println("legendre(0.5)=",legendre(0.5))

dlr = DLR.DLRGrid(:fermi, 10*EF,β, 1e-2)

const qgrid = Grid.boseKUL(kF, 10kF, 0.000001*sqrt(me^2/β/kF^2), 15,4) 
const τgrid = Grid.tauUL(β, 0.0001, 11,4)
const vqinv = [(q^2 + mass2) / (4π * e0^2) for q in qgrid.grid]
# println(qgrid.grid)
# println(τgrid.grid)

const dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, me) # dynamic part of the effective interaction

const NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
println("NF=$NF")

function lindhard(x)
    if (abs(x) < 1.0e-8)
        return 1.0
    elseif (abs(x - 1.0) < 1.0e-8)
        return 0.5
    else
        return 0.5 - (x^2 - 1) / 4.0 / x * log(abs((1 + x) / (1 - x)))
    end
end

function interaction(q, τIn, τOut)
    dτ = abs(τOut - τIn)

    kQ = sqrt(dot(q, q))
    v = 4π * e0^2 / (kQ^2 + mass2)
    if kQ <= qgrid.grid[1]
        w = v * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-14, dτ) # the current interpolation vanishes at q=0, which needs to be corrected!
    else
        w = v * Grid.linear2D(dW0, qgrid, τgrid, kQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end
    v = 4π * e0^2 / (kQ^2 + mass2 + 4π * e0^2 * NF * lindhard(kQ / 2.0 / kF))
    v = v/β - w
#    v = v/β
    return v, w
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
    result = 0.0+0.0im

    if config.curr == 1
        T,K,N,Theta = config.var[1],config.var[2], config.var[3],config.var[4]
        n1,n2 = dlr.n[N[1]],dlr.n[N[2]]
        q = K[1]
        t1, t2, t3 = T[1], T[2], T[3]
        θ = Theta[1]
        θ = abs(π-θ)

        k1 = 0.9kF * @SVector[1, 0, 0]
        k2 = 1.1kF * @SVector[cos(θ), sin(θ), 0]

        ω1 = (dot(q-k1, q-k1) - kF^2) * β

        ω2 = (dot(q-k2, q-k2) - kF^2) * β

        τ1 = (-t3)/β
        g1 = Spectral.kernelFermiT(τ1, ω1)

        τ3 = (-t1)/β
        g3 = Spectral.kernelFermiT(τ3, ω1)

        τ2 = (t1-t2)/β
        g2 = Spectral.kernelFermiT(τ2, ω2)

        τ4 = (t1)/β
        g4 = Spectral.kernelFermiT(τ4, ω2)

        W1 = interaction(q-k1-k2, 0.0, t2)
        W2 = interaction(q, (t1),(t3) )

        factor = 1.0/(2π)^3 * legendre(cos(θ))*sin(θ)/2.0
        r_r = W1[2]*W2[2]*g1*g2* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (t2-t3)/β)
        # s_r = W1[1]*W2[2]*g1*g2/β
        # r_s = W1[2]*W2[1]*g1*g2/β
        # s_s = W1[1]*W2[1]*g1*g2
        s_r = W1[1]*W2[2]*g1*g4* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (-t3)/β)
        r_s = W1[2]*W2[1]*g2*g3* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (t2-t1)/β)
        s_s = W1[1]*W2[1]*g3*g4* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (-t1)/β)

        result =  (s_s+s_r+r_s+r_r)

        τ1 = (-t3)/β
        g1 = Spectral.kernelFermiT(τ1, ω1)

        τ2 = (t3-t2)/β
        g2 = Spectral.kernelFermiT(τ2, ω2)

        τ3 = (-t1)/β
        g3 = Spectral.kernelFermiT(τ3, ω1)

        τ4 = (t3)/β
        g4 = Spectral.kernelFermiT(τ4, ω2)

        τ6 = (t1-t2)/β
        g6 = Spectral.kernelFermiT(τ6, ω2)

        τ8 = (t1)/β
        g8 = Spectral.kernelFermiT(τ4, ω2)

        W1 = interaction(q, 0.0, t2)
        W2 = interaction(k1-k2, (t1),(t3) )

        factor = 1.0/(2π)^3 * legendre(cos(θ))*sin(θ)/2.0
        r_r = W1[2]*W2[2]*g1*g2* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (t2-t1)/β)
        # s_r = W1[1]*W2[2]*g1*g2/β
        # r_s = W1[2]*W2[1]*g1*g2/β
        # s_s = W1[1]*W2[1]*g1*g2
        s_r = W1[1]*W2[2]*g1*g4* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (-t1)/β)
        r_s = W1[2]*W2[1]*g3*g6* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (t2-t1)/β)
        s_s = W1[1]*W2[1]*g3*g8* factor*exp(im*π*(2*n1+1) * t1 /β) * exp(im*π*(2*n2+1) * (-t1)/β)

        result +=  (s_s+s_r+r_s+r_r) * 2.0
    else
        result =  0.0+0.0*im
    end
    return Weight(real(result),imag(result))
end


function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    obs[config.var[3][1],config.var[3][2],:] += weight / abs(weight) * factor

end

function run(steps)

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    N = MonteCarlo.Discrete(1, length(dlr.n))
    Theta = MonteCarlo.Angle()

    dof = [[3,1,2,1],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64,(length(dlr.n),length(dlr.n),2))

    #    nei = [[2,4],[3,1],[2,4],[1,3]]

    config = MonteCarlo.Configuration(steps, (T,K,N,Theta ), dof, obs)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=4)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    # NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
    # println("NF=$NF")


    if isnothing(avg) == false

        println("fourier")
        avg_c = avg[:,:,1]+im*avg[:,:,2]
        avg_tw = (DLR.matfreq2tau(:fermi,avg_c,dlr,dlr.τ;axis=1,rtol=1e-2))
        avg_tt = (DLR.matfreq2tau(:fermi,avg_tw,dlr,dlr.τ;axis=2,rtol=1e-2))

        avg_wt = (DLR.tau2matfreq(:fermi,avg_tt,dlr,dlr.n;axis=1,rtol=1e-2))
        avg_ww = (DLR.tau2matfreq(:fermi,avg_wt,dlr,dlr.n;axis=2,rtol=1e-2))
        println(size(avg_ww))

        for i in 1:length(dlr.n)
            for j in 1:length(dlr.n)
                @printf("%d\t %d\t %10.6f ± %10.6f\t %10.6f\n",
                        dlr.n[i],dlr.n[j], (avg[i,j,1]), (std[i,j,1]), real(avg_ww[i,j]))
            end
        end

        # println("tau-tau")
        # for i in 1:length(dlr.n)
        #     for j in 1:length(dlr.n)
        #         @printf("%10.6f\t %10.6f\t %10.6f\t%10.6f\n",
        #                 dlr.τ[i],dlr.τ[j], real(avg_tt[i,j]),imag(avg_tt[i,j]))
        #     end
        # end
    end
end

run(Steps)
# @time run(Steps)
