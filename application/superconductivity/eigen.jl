using StaticArrays:similar
using QuantumStatistics
using FastGaussQuadrature
using Printf
include("parameter.jl")



    ##### Panels endpoints for composite quadrature rule ###
# npo = Int(ceil(log(β * Euv) / log(2.0)))
# pbp = zeros(Float64, 2npo + 1)
# pbp[npo + 1] = 0.0
# for i in 1:npo
#     pbp[npo + i + 1] = 1.0 / 2^(npo - i)
# end
# pbp[1:npo] = -pbp[2npo + 1:-1:npo + 2]

# function Green(n, IsMatFreq)
#         # n: polynomial order
#     xl, wl = gausslegendre(n)
#     xj, wj = gaussjacobi(n, 1 / 2, 0.0)

#     G = IsMatFreq ? zeros(ComplexF64, length(Grid)) : zeros(Float64, length(Grid))
#         err = zeros(Float64, length(Grid))
#         for (τi, τ) in enumerate(Grid)
#             for ii in 2:2npo - 1
#                 a, b = pbp[ii], pbp[ii + 1]
#                 for jj in 1:n
#                     x = (a + b) / 2 + (b - a) / 2 * xl[jj]
#                     if type == :corr && x < 0.0 
#                         # spectral density is defined for positivie frequency only for correlation functions
#                         continue
#                     end
#                     ker = IsMatFreq ? Spectral.kernelΩ(type, τ, Euv * x, β) : Spectral.kernelT(type, τ, Euv * x, β)
#                     G[τi] += (b - a) / 2 * wl[jj] * ker * sqrt(1 - x^2)
#             end

function KGrid(Order, Nk)
    maxK = 10.0
    minK = 0.001
    #Nk = 10
    #Order = 8 # Legendre polynomial order
    panel = Grid.boseKUL(0.5 * kF, maxK, minK, Nk, 1).grid
    panel[1] = 0.0  # the kgrid start with 0.0
    println("Panels Num: ", length(panel))

    # n: polynomial order
    x, w = gausslegendre(Order)
    println("Guassian quadrature points : ", x)
    println("Guassian quadrature weights: ", w)

    println("KGrid Num: ", (length(panel)-1) * Order )

    kgrid = zeros(Float64, (length(panel)-1) * Order )
    wkgrid = similar(kgrid)

    for pidx in 1:length(panel) - 1
        a, b = panel[pidx], panel[pidx + 1]
        for o in 1:Order
            idx = (pidx - 1) * Order + o
            kgrid[idx] = (a + b) / 2 + (b - a) / 2 * x[o]
            @assert kgrid[idx] > a && kgrid[idx] < b
            wkgrid[idx] = (b - a) / 2 * w[o]
        end
    end
    #kgrid[length(kgrid)]=maxK
    return kgrid, wkgrid
end


function KGrid_bose(Order, Nk)
    maxK = 10.0
    minK = 0.001
    #Nk = 10
    #Order = 8 # Legendre polynomial order
    panel = Grid.boseKUL(0.5 * kF, maxK, minK, Nk, 1).grid
    panel[1] = 0.0  # the kgrid start with 0.0
    println("Panels Num: ", length(panel))

    # n: polynomial order
    x, w = gausslegendre(Order)
    println("Guassian quadrature points : ", x)
    println("Guassian quadrature weights: ", w)

    println("KGrid Num: ", (length(panel)-1) * Order )

    kgrid = zeros(Float64, (length(panel)-1) * Order )
    wkgrid = similar(kgrid)

    for pidx in 1:length(panel) - 1
        a, b = panel[pidx], panel[pidx + 1]
        for o in 1:Order
            idx = (pidx - 1) * Order + o
            kgrid[idx] = (a + b) / 2 + (b - a) / 2 * x[o]
            @assert kgrid[idx] > a && kgrid[idx] < b
            wkgrid[idx] = (b - a) / 2 * w[o]
        end
    end
    #kgrid[length(kgrid)]=maxK
    return kgrid, wkgrid
end


function build_int(k, q, kbose_grid)
    result=FLoat64[]
    push!(result, abs(k-q))
    for (pi, p) in enumerate(kbose_grid)
        if(p>abs(k-q)&&p<k+q)
            push!(result, p)
        end
    end
    push!(result, k+q)
    return result
end

function legendre_dc(kernal_bose, fdlr, kgrid ,qgrid, kbose_grid)
    kernal = zeros(FLoat64, (fdlr.size, length(kgrid), length(kgrid)))
    for (ki, k) in enumerate(kgrid)
        for (qi, q) in enumerate(qgrid[ki])
            mom_int, w_mom_int=build_int(k, q, kbose_grid)
            
        end
    end
end



"""
calcF(Δ, kgrid, fdlr)

calculate the F function in τ-k representation 

# Arguments:

- Δ: gap function in (ωn, k) representation
- kgrid: momentum grid
- fdlr::DLRGrid: DLR Grid that contains the imaginary-time grid
"""
function calcF(Δ0, Δ, fdlr, kgrid)
    Δ = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=1)
    F = zeros(ComplexF64, (fdlr.size, length(kgrid)))
    for (ki, k) in enumerate(kgrid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            F[ni, ki] = (Δ[ni, ki] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    end
end
                F = DLR.matfreq2tau(:fermi, F, fdlr, fdlr.τ, axis=1)
    return real(F)
end

function dH1(k, p, τ)
    g = e0^2
    gh = sqrt(g)
    if abs(k - p) > 1.0e-12
        return -2π * gh^3 * (log((abs(k - p) + gh)) -log( abs(k - p))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
        #return -2π * gh^3 * (log((abs(k - p) + gh))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
    else
        return 0.0
end
end

function bare(k, p)
    if abs(k - p) > 1.0e-12
        return 4π * e0^2 * (log(k + p) - log(abs(k - p)))
        #return 4π * e0^2 * log(k + p)         
    else
        return 0.0
end
end

function calcΔ(F, fdlr, kgrid, qgrid, wkgrid)
    Δ0 = zeros(Float64, length(kgrid))
    Δ = zeros(Float64, (fdlr.size, length(kgrid)))
    for (ki, k) in enumerate(kgrid)
        for (qi, q) in enumerate(qgrid)
            Δ0[ki] += bare(k, q) * F[1, qi] * wkgrid[qi]  # TODO: check -F(0^-, q)==F(0^+, q)
            for (τi, τ) in enumerate(fdlr.τ)
                # Δ[τi, ki] += dW[τi, qi, ki] * F[τi, qi] * wkgrid[qi]
                Δ[τi, ki] += dH1(k, q, τ) * F[τi, qi] * wkgrid[qi]
            end
        end
    end
    return Δ0, Δ 
end

if abspath(PROGRAM_FILE) == @__FILE__
    # KGrid()
    fdlr = DLR.DLRGrid(:fermi, 10EF, β, 1e-10) 
                
    ########## non-uniform kgrid #############
    kgrid, wkgrid = KGrid(64,10)
    kgrid_double, wkgrid_double = KGrid(32,10)
    println(sum(wkgrid))
    ########## uniform K grid  ##############
    # MaxK = 5.0 * kF
    # Nk = 256
    # dK = MaxK / Nk
    # kgrid = LinRange(0.0, MaxK, Nk) .+ dK / 2.0
    # wkgrid = [Nk / MaxK for i in 1:Nk]

    Δ = zeros(Float64, (fdlr.size, length(kgrid)))
    Δ0 = zeros(Float64, length(kgrid)) .+ 1.0
    F = calcF(Δ0, Δ, fdlr, kgrid)
    filename = "./bare.dat"
    open(filename, "w") do io
        for r in 1:length(kgrid)
            @printf(io, "%32.17g  %32.17g %32.17g %32.17g\n",kgrid[r] , bare(2.0, kgrid[r]) * F[1, r]*wkgrid[r], bare(2.0, kgrid[r])*wkgrid[r], F[1, r]*wkgrid[r])
        end
    end
    
    println("sumF",sum(F),maximum(abs.(F)))
    Δ0, Δ = calcΔ(F, fdlr, kgrid, kgrid, wkgrid)
    Δ_freq = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=1)

    Δ_2 = zeros(Float64, (fdlr.size, length(kgrid_double)))
    Δ0_2 = zeros(Float64, length(kgrid_double)) .+ 1.0
    F = calcF(Δ0_2, Δ_2, fdlr, kgrid_double)
    println(sum(F))
    Δ0_double, Δ_double = calcΔ(F, fdlr, kgrid, kgrid_double, wkgrid_double)
    #Δ_freq = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=1)
    #maxvl, maxindex= findmax(reshape(abs.(Δ-Δ_double), fdlr.size*length(kgrid)))
    #println(maxindex)
    #index1 = maxindex ÷
    maxindex=1
    for r in 1:length(kgrid)
        if kgrid[r]<2.0
            global maxindex=r
        end
    end
    println(abs.(Δ0-Δ0_double)[maxindex],",", kgrid[maxindex])
    #println(abs.(Δ-Δ_double)[maxindex],",", kgrid[maxindex])
    filename = "./test.dat"
    println(fdlr.n,fdlr.n[fdlr.size÷2+1])
    open(filename, "w") do io
        for r in 1:length(kgrid)
            @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid[r] ,Δ0[r] ,real(Δ_freq[fdlr.size÷2+1,r]))
        end
    end
    # filename = "./.dat"    
    # open(filename, "w") do io
    #     for r in 1:length(kgrid)
    #         @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid[r] ,Δ0[r] ,real(Δ_freq[fdlr.size÷2+1,r]))
    #     end
    # end
end
