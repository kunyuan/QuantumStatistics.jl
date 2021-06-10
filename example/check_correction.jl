
# calculate (d) diagram in Kohn-Luttinger's paper as kernel of gap-equation at kF on dlr tau grids

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters, Dierckx
# using ProfileView
using PyCall
const Steps = 1e7

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

#
# G(w,k)G(-w,-k)
#

function GG(ωin, ω, β)
    return 1.0/(ωin ^2 + ω^2 )
end

#
# gap-function
#

dataFileName = "../../data_from_t/data1/delta_56.dat"

f = open(dataFileName, "r")

MomBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
FreqBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
data = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])

data = reshape(data,(length(MomBin),length(FreqBin)))

delta_spl = Spline2D(MomBin,FreqBin,data)

const lamu = 0.2229769140136642

function Δ(ω, k, isNormed = true)
    if ω<0
        ω=-ω
    end
    if ω/kF^2<0.25 && isNormed
        return delta_spl(k/kF,ω/kF^2)*lamu#GG(n,(k^2-kF^2),β)
    else
        return delta_spl(k/kF,ω/kF^2)

    end
end

println(delta_spl(0.999, 0.001))

#ExtFreqBin = FreqBin[1:40]*kF^2
ExtFreqBin = [π*(2*dlr.n[i]+1)/β for i in 1:length(dlr.n)][17:end-3]

#
# interaction
#
const kgrid = Grid.fermiKUL(kF, 9kF, 0.00001kF, 4,4) 
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
    kin, kout = 0.9kF, 1.1kF

    if config.curr == 2
        T,K,Ext1,Ext2,Theta,K2,N2 = config.var[1],config.var[2], config.var[3],config.var[4],config.var[5],config.var[6],config.var[7]
        n1,n2 = Ext1[1],N2[1]
        ωout, ωin = ExtFreqBin[n1], π*(2n2+1)/β
        q = K[1]
        t1, t2, t3 = T[1], T[2], T[3]
        θ = Theta[1]
        θ = abs(π-θ)

        k1 = kgrid[Ext2[1]] * @SVector[1, 0, 0]
        k2 = K2[1] * @SVector[cos(θ), sin(θ), 0]

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

        ω0 = (dot(k2,k2)-kF^2)
        factor = -1.0/(2π)^3/(2π)^2/β * legendre(cos(θ))*sin(θ)/2.0 * GG(ωin, ω0,β) * Δ(ωin,K2[1]) * kgrid[Ext2[1]]

        r_r = W1[2]*W2[2]*g1*g2* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (t2-t3))
        s_r = W1[1]*W2[2]*g1*g4* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (-t3))
        r_s = W1[2]*W2[1]*g2*g3* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (t2-t1))
        s_s = W1[1]*W2[1]*g3*g4* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (-t1))

        result += (s_s+s_r+r_s+r_r)

        # ω1 = (dot(q-k1, q-k1) - kF^2) * β

        # ω2 = (dot(q-k2, q-k2) - kF^2) * β

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

        ω0 = (dot(k2,k2)-kF^2)

        r_r = W1[2]*W2[2]*g1*g2* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (t2-t1)/β)
        s_r = W1[1]*W2[2]*g1*g4* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (-t1)/β)
        r_s = W1[2]*W2[1]*g3*g6* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (t2-t1)/β)
        s_s = W1[1]*W2[1]*g3*g8* factor*exp(im*ωout * t1 /β) * 2.0*cos(ωin * (-t1)/β)

        result +=  (s_s+s_r+r_s+r_r) * 2.0

    # elseif config.curr == 2
    # bare interaction
    elseif config.curr==1
        T,K,Ext1,Ext2,Theta,K2,N2 = config.var[1],config.var[2], config.var[3],config.var[4],config.var[5],config.var[6],config.var[7]
        n1,n2 = Ext1[1],N2[1]
        ωout, ωin = ExtFreqBin[n1], π*(2n2+1)/β
        t1 = T[1]
        θ = Theta[1]
        θ = abs(π-θ)

        k1 = kgrid[Ext2[1]] * @SVector[1, 0, 0]
        k2 = K2[1] * @SVector[cos(θ), sin(θ), 0]

        W1 = interaction(k1-k2, 0, t1)

        ω0 = (dot(k2,k2)-kF^2)
        factor =  -1.0/(2π)^2/β * legendre(cos(θ))*sin(θ)/2.0 * GG(ωin, ω0,β) * Δ(ωin,K2[1]) * kgrid[Ext2[1]]
        r_0 = W1[2] * factor * exp(im*ωout * t1 ) * 2.0*cos(ωin * (-t1))
        s_0 = W1[1] * factor * 2.0

        result += r_0 + s_0
    else
        result =  0.0+0.0*im
    end
    return Weight(real(result),imag(result))
end


function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    obs[config.var[3][1],config.var[4][1],:] += weight / abs(weight) * factor

end

function run(steps)

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext1 = MonteCarlo.Discrete(1, length(ExtFreqBin))
    Ext2 = MonteCarlo.Discrete(1, kgrid.size)
    Theta = MonteCarlo.Angle()
    K2 = MonteCarlo.Tau(MomBin[end], kF)
    N2 = MonteCarlo.Discrete(0, floor(Int, 5EF/(2π/β)))

#    dof = [[1,0,1,1,1,1,1],] # degrees of freedom of the normalization diagram and the bubble
    dof = [[1,0,1,1,1,1,1],[3,1,1,1,1,1,1]] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[3,1,1,1,1,1,1],] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[3,1,1,1,1,1,1],[1,0,1,1,1,1,1]] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64,(length(ExtFreqBin),kgrid.size,2))

    #    nei = [[2,4],[3,1],[2,4],[1,3]]

    config = MonteCarlo.Configuration(steps, (T,K,Ext1,Ext2,Theta,K2,N2), dof, obs)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    # NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
    # println("NF=$NF")


    if isnothing(avg) == false
        Δ_ext = zeros(Float64,(length(ExtFreqBin),kgrid.size))
        for i in 1:length(ExtFreqBin)
            for j in 1:kgrid.size
                Δ_ext[i,j] = Δ(ExtFreqBin[i],kgrid[j],false)
            end
        end
        norm = sum( abs.(avg[:,:,1]))/sum( abs.(Δ_ext) )
        println("norm=$norm")
        norm = 1.0
        avg, std = avg/norm, std/norm
        for i in 1:length(ExtFreqBin)
            for j in 1:kgrid.size
                @printf("%10.6f\t %10.6f\t %10.6f ± %10.6f \t %10.6f\n",
                        ExtFreqBin[i],kgrid[j],
                        (avg[i,j,1]), (std[i,j,1]),#(avg[i,j,2]), (std[i,j,2]),
                        Δ_ext[i,j])
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
