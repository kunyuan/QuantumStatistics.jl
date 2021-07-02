using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters, Dierckx
# using ProfileView

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

push!(LOAD_PATH, rundir)
using parameter
println("rs=$rs, β=$β, kF=$kF, EF=$EF, mass2=$mass2")

include(srcdir*"/eigen.jl")
#
# G(w,k)G(-w,-k)
#

dataFileName = rundir*"/sigma.dat"

f = open(dataFileName, "r")

readline(f)
MomBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
FreqBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
data1 = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
data2 = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])

#data1,data2 = reshape(data1,(length(MomBin),length(FreqBin))),reshape(data2,(length(MomBin),length(FreqBin)))
# sigmaR_spl = Spline2D(MomBin,FreqBin,data1)
# sigmaI_spl = Spline2D(MomBin,FreqBin,data2)

data1,data2 = reshape(data1,(length(FreqBin),length(MomBin))),reshape(data2,(length(FreqBin),length(MomBin)))
sigmaR_spl = Spline2D(FreqBin,MomBin,data1;kx=1,ky=1,s=0.0)
sigmaI_spl = Spline2D(FreqBin,MomBin,data2;kx=1,ky=1,s=0.0)

println("ΣR=",sigmaR_spl(0.0, 1.0),"\tΣI=",sigmaI_spl(0.0, 1.0))
println("ΣR=",sigmaR_spl(1.0, 0.0),"\tΣI=",sigmaI_spl(1.0, 0.0))

function GG(ωin, k, β=1.0)
    ω = k^2-kF^2
    factor = kF^2
    # ΣR = sigmaR_spl(k/kF, ωin/kF^2)*factor
    # ΣI = sigmaI_spl(k/kF, ωin/kF^2)*factor
    ΣR = sigmaR_spl( ωin/kF^2,k/kF)*factor
    ΣI = sigmaI_spl( ωin/kF^2,k/kF)*factor
    return 1.0/((ωin-ΣI) ^2 + (ω+ΣR)^2 )
end

#
# gap-function
#

dataFileName = rundir*"/delta.dat"

f = open(dataFileName, "r")

MomBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
FreqBin = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])
data = map(x->parse(Float64,x),split(readline(f)," ")[1:end-1])

data = reshape(data,(length(MomBin),length(FreqBin)))

delta_spl = Spline2D(MomBin,FreqBin,data;kx=1,ky=1,s=0.0)

const lamu = 0.2049458542291443 #beta200, rs4
#const lamu = 0.2229769140136642 #beta250, rs4
#const lamu = 0.002398499678459926 #beta25, rs2
#const lamu = 0.02231536377399141 #beta250,rs2

const freq_cut = 0.25
const actual_cut_index = findmin([(freq-freq_cut<0 ? FreqBin[end] : freq ) for freq in FreqBin ])[2]-1

function Δ(ω, k, isNormed = false)
    if ω<0
        ω=-ω
    end
    cut = freq_cut
#   cut = FreqBin[actual_cut_index]
    if ω/kF^2<=cut && isNormed
        return delta_spl(k/kF,ω/kF^2)*lamu
    else
        return delta_spl(k/kF,ω/kF^2)

    end
end

println(delta_spl(0.999, 0.001))

if abspath(PROGRAM_FILE) == @__FILE__
    fdlr = DLR.DLRGrid(:acorr, 100EF, β, 1e-10)
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 
    ########## non-uniform kgrid #############
    Nk = 16
    order = 8
    maxK = 10.0 * kF
    minK = 0.00001 / (β * kF)
    #minK = 0.0000001    
    kpanel = KPanel(Nk, kF, maxK, minK)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    println("kgrid number: $(length(kgrid.grid))")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))

    Δ_freq = zeros(Float64, (length(kgrid.grid), fdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        for(ωi, ω) in enumerate(fdlr.ω)
            Δ_freq[ki, ωi] = Δ(ω, k)
        end
    end

    F_freq = zeros(ComplexF64, (length(kgrid.grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            #F_freq[ki, ni] = (Δ_freq[ki, ni]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            F_freq[ki, ni] = (Δ_freq[ki, ni]) * GG(π*(2n+1)/β, k)
            #F[ki, ni] = (Δ[ki, ni]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            
        end

    end

    println("F_freq_max:", maximum(abs.(F_freq)))
    #println("F_imag=$(F_max)")
    F_τ = real(DLR.matfreq2tau(:acorr, F_freq, fdlr, extT_grid.grid, axis=2))
    println("F_τ_max:",maximum(F_τ))
    F_ext = zeros(Float64, (length(extK_grid.grid), length(extT_grid.grid)))

    outFileName = rundir*"/f.dat"
    f = open(outFileName, "w")

    kpidx = 1
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    x = @view kgrid.grid[head:tail]
    w = @view kgrid.wgrid[head:tail]
    for (ki, k) in enumerate(extK_grid.grid)
        if k > kgrid.panel[kpidx + 1]
            # if q is larger than the end of the current panel, move k panel to the next panel
            while k > kgrid.panel[kpidx + 1]
                global kpidx += 1
            end
            global head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            global x = @view kgrid.grid[head:tail]
            global w = @view kgrid.wgrid[head:tail]
            @assert kpidx <= kgrid.Np
        end
        for (τi, τ) in enumerate(extT_grid.grid)
            fx = @view F_τ[head:tail, τi] # all F in the same kpidx-th K panel
            F_ext[ki, τi] = barycheb(order, k, fx, w, x) # the interpolation is independent with the panel length
            #@printf("%32.17g  %32.17g  %32.17g\n",extK_grid[ki] ,extT_grid[τi], F_ext[ki, τi])
            @printf(f, "%32.17g  %32.17g  %32.17g\n",extK_grid[ki] ,extT_grid[τi], F_ext[ki, τi])
        end
    end


end
