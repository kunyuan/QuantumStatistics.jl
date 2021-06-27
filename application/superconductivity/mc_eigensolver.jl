
# calculate (d) diagram in Kohn-Luttinger's paper as kernel of gap-equation at kF on dlr tau grids

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters, Dierckx
# using ProfileView
using PyCall
const Steps = 4e6

include("parameter.jl")
include("eigen.jl")
include("grid.jl")
#include("../application/electron_gas/RPA.jl")

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
#const kgrid = Grid.fermiKUL(kF, 9kF, 0.001kF, 3,4) 
#const kgrid = Grid.Uniform{Float64, 2}(kF, 1.001kF,[false, false])
const Nk = 16
const order = 8
const maxK = 10.0 * kF
const minK = 0.00001 / (β * kF)
const kpanel = KPanel(Nk, kF, maxK, minK)
const kgrid = CompositeGrid(kpanel, order, :cheb)
fdlr = DLR.DLRGrid(:fermi, 100EF, β, 1e-10)


#
# G(w,k)G(-w,-k)
#

#dataFileName = "/home/xc82a/data_from_t/s_largeT/sigma_0.dat"
# dataFileName = "/home/xc82a/data_from_t/s/sigma_96.dat"

# f = open(dataFileName, "r")

# readline(f)
# MomBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
# FreqBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
# data1 = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
# data2 = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])

# #data1,data2 = reshape(data1,(length(MomBin),length(FreqBin))),reshape(data2,(length(MomBin),length(FreqBin)))
# # sigmaR_spl = Spline2D(MomBin,FreqBin,data1)
# # sigmaI_spl = Spline2D(MomBin,FreqBin,data2)

# data1,data2 = reshape(data1,(length(FreqBin),length(MomBin))),reshape(data2,(length(FreqBin),length(MomBin)))
# sigmaR_spl = Spline2D(FreqBin,MomBin,data1;kx=1,ky=1,s=0.0)
# sigmaI_spl = Spline2D(FreqBin,MomBin,data2;kx=1,ky=1,s=0.0)

# println("ΣR=",sigmaR_spl(0.0, 1.0),"\tΣI=",sigmaI_spl(0.0, 1.0))
#println("ΣR=",sigmaR_spl(1.0, 0.0),"\tΣI=",sigmaI_spl(1.0, 0.0))

# for i in 1:kgrid.size
#     @printf("%10.6f\t %10.6f\n",
#             kgrid[i]/kF,sigmaR_spl(0.0, kgrid[i]/kF))
# end

function GG(ωin, k, β=1.0)
    ω = k^2-kF^2
    factor = kF^2
    # ΣR = sigmaR_spl(k/kF, ωin/kF^2)*factor
    # ΣI = sigmaI_spl(k/kF, ωin/kF^2)*factor
    # ΣR = sigmaR_spl( ωin/kF^2,k/kF)*factor
    # ΣI = sigmaI_spl( ωin/kF^2,k/kF)*factor
    ΣR, ΣI = 0.0, 0.0
    return 1.0/((ωin-ΣI) ^2 + (ω+ΣR)^2 )
end

#
# gap-function
#

#dataFileName = "/home/xc82a/data_from_t/s_largeT/delta_0.dat"
# dataFileName = "/home/xc82a/data_from_t/s/delta_96.dat"

# f = open(dataFileName, "r")

# MomBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
# FreqBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
# data = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])

# data = reshape(data,(length(MomBin),length(FreqBin)))

# delta_spl = Spline2D(MomBin,FreqBin,data;kx=1,ky=1,s=0.0)

# const lamu = 0.2229769140136642 #beta250, rs4
# #const lamu = 0.002398499678459926 #beta25, rs2
# #const lamu = 0.02231536377399141 #beta250,rs2

# const freq_cut = 0.25
# const actual_cut_index = findmin([(freq-freq_cut<0 ? FreqBin[end] : freq ) for freq in FreqBin ])[2]-1

function Δ0(k)
    return 1.0
end

function Δ(τ, k)
    if τ<0
        τ=-τ
    end
    return 1.0
end

#println(delta_spl(0.999, 0.001))

#ExtFreqBin = FreqBin[1:9]*kF^2
half = Base.floor(Int, length(dlr.n)/2)
ExtFreqBin = [π*(2*dlr.n[i]+1)/β for i in 1:length(dlr.n)][half+1:half+2]
#ExtFreqBin = [π*(2*dlr.n[i]+1)/β for i in 1:length(dlr.n)]

#
# interaction
#

const qgrid = Grid.boseKUL(kF, 10kF, 0.00001*sqrt(me^2/β/kF^2), 12,8) 
const τgrid = Grid.tauUL(β, 0.0001, 11,8)
const vqinv = [(q^2 + mass2) / (4π * e0^2) for q in qgrid.grid]
# println(qgrid.grid)
# println(τgrid.grid)

#const dW0 = dWRPA(vqinv, qgrid.grid, τgrid.grid, kF, β, spin, me) # dynamic part of the effective interaction

#const NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
#println("NF=$NF")

function lindhard(x)
    if (abs(x) < 1.0e-8)
        return 1.0
    elseif (abs(x - 1.0) < 1.0e-8)
        return 0.5
    else
        return 0.5 - (x^2 - 1) / 4.0 / x * log(abs((1 + x) / (1 - x)))
    end
end


"""
   W_art(τ, q)

An artificial interaction that mimic RPA and can be treated analytically.
Return bare interaction v/β and dynamical part w.
The interaction in frequency space would be:
v = 4π * g / q^2
w = -4π * g / q^2 * g*kF^3 / ( ω^2 + q^2*kF^2 + g*kF^3 )

#Arguments:
    - τ: τ
    - q: q
"""
function W_art(τ, q, g = e0^2, β = β)
    q2 = dot(q,q)
    sq2g=sqrt(q2*kF^2+g*kF^3)
    v = 4π * g / (q2)
    w = -2*π*g^2*kF^3/(q2)/sq2g*(exp(-sq2g*τ)+exp(-sq2g*(β-τ)))/(1-exp(-sq2g*β))
    return v/β, w
end

function interaction(q, τIn, τOut)
    dτ = abs(τOut - τIn)
    τ = dτ

#    kQ = sqrt(dot(q, q))
    # q2 = dot(q,q)
    # g = e0^2
    # sq2g=sqrt(q2*kF^2+g*kF^3)
    # v = 4π * e0^2 / (q2 + mass2)
    # if kQ <= qgrid.grid[1]
    #     w = v * Grid.linear2D(dW0, qgrid, τgrid, qgrid.grid[1] + 1.0e-14, dτ) # the current interpolation vanishes at q=0, which needs to be corrected!
    # else
    #     w = v * Grid.linear2D(dW0, qgrid, τgrid, kQ, dτ) # dynamic interaction, don't forget the singular factor vq
    # end
    # v = 4π * e0^2 / (kQ^2 + mass2 + 4π * e0^2 * NF * lindhard(kQ / 2.0 / kF))
    # v = v/β - w
    #    v = v/β
#    w = -2*π*g^2*kF^3/(q2)/sq2g*(exp(-sq2g*τ)+exp(-sq2g*(β-τ)))/(1-exp(-sq2g*β))
    return W_art(τ, q)
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

    # bare interaction
    if config.curr==1
        T,Ext1,Ext2,Theta,K2 = config.var[1],config.var[2], config.var[3],config.var[4],config.var[5]
        n1 = Ext1[1]
        ωout = ExtFreqBin[n1]
        t1, t2 = T[1], T[2]
        θ = Theta[1]
        θ = abs(π-θ)

        k1 = kgrid[Ext2[1]] * @SVector[1, 0, 0]
        k2 = K2[1] * @SVector[cos(θ), sin(θ), 0]

        W1 = interaction(k1-k2, 0, t1)

        τ1 = (t2-t1)/β
        ω2 = (dot(k2, k2) - kF^2) * β
        gg = Spectral.kernelFermiT(τ1, ω2)*Spectral.kernelFermiT(t2/β, ω2)
        gg0 = Spectral.kernelFermiT(t2/β, ω2)*Spectral.kernelFermiT(t2/β, ω2)

        factor =  -1.0/(2π)^2 * legendre(cos(θ))*sin(θ)/2.0 * kgrid[Ext2[1]]*K2[1] * Δ0(K2[1])
        r_0 = W1[2] * factor * exp(im*ωout * t1 ) * gg
        s_0 = W1[1] * factor * gg0

        result += r_0 + s_0

    elseif config.curr==2
        T,Ext1,Ext2,Theta,K2 = config.var[1],config.var[2], config.var[3],config.var[4],config.var[5]
        n1 = Ext1[1]
        ωout = ExtFreqBin[n1]
        t1, t2, t3 = T[1], T[2], T[3]
        θ = Theta[1]
        θ = abs(π-θ)

        k1 = kgrid[Ext2[1]] * @SVector[1, 0, 0]
        k2 = K2[1] * @SVector[cos(θ), sin(θ), 0]

        W1 = interaction(k1-k2, 0, t1)

        τ1 = (t2-t1)/β
        τ2 = t3/β
        ω2 = (dot(k2, k2) - kF^2) * β
        gg = Spectral.kernelFermiT(τ1, ω2)*Spectral.kernelFermiT(τ2, ω2)
        gg0 = Spectral.kernelFermiT(t2/β, ω2)*Spectral.kernelFermiT(t3/β, ω2)

        factor =  -1.0/(2π)^2 * legendre(cos(θ))*sin(θ)/2.0 * kgrid[Ext2[1]]*K2[1] * Δ(t3-t2, K2[1])
        r_0 = W1[2] * factor * exp(im*ωout * t1 ) * gg
        s_0 = W1[1] * factor * gg0

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
    obs[config.var[2][1],config.var[3][1],:] += weight / abs(weight) * factor

end

function run(steps)

    T = MonteCarlo.Tau(β, β / 2.0)
    K = MonteCarlo.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext1 = MonteCarlo.Discrete(1, length(ExtFreqBin))
    Ext2 = MonteCarlo.Discrete(1, kgrid.size)
    Theta = MonteCarlo.Angle()
    K2 = MonteCarlo.Tau(10*kF, kF)

    dof = [[2,1,1,1,1],] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[2,1,1,1,1],[3,1,1,1,1]] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[1,0,1,1,1,1,1],[3,1,1,1,1,1,1]] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[3,1,1,1,1,1,1],] # degrees of freedom of the normalization diagram and the bubble
#    dof = [[3,1,1,1,1,1,1],[1,0,1,1,1,1,1]] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64,(length(ExtFreqBin),kgrid.size,2))

    #    nei = [[2,4],[3,1],[2,4],[1,3]]

    config = MonteCarlo.Configuration(steps, (T,Ext1,Ext2,Theta,K2), dof, obs)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    # NF = -TwoPoint.LindhardΩnFiniteTemperature(dim, 0.0, 0, kF, β, me, spin)[1]
    # println("NF=$NF")


    if isnothing(avg) == false
        Δ_ext = zeros(Float64,(length(ExtFreqBin),kgrid.size))
        for i in 1:length(ExtFreqBin)
            for j in 1:kgrid.size
                Δ_ext[i,j] = Δ(ExtFreqBin[i],kgrid[j])
            end
        end
        norm = sum( abs.(avg[:,:,1]))/sum( abs.(Δ_ext) )
        println("norm=$norm")
        norm = 1.0
        avg, std = avg/norm, std/norm
        for i in 1:length(ExtFreqBin)
            for j in 1:kgrid.size
                @printf("%10.8f\t %10.8f\t %10.8f ± %10.8f \t %10.8f\n",
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
