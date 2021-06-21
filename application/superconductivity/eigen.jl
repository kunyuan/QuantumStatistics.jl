using StaticArrays:similar, maximum
using QuantumStatistics
using Printf
using Plots
include("parameter.jl")
include("grid.jl")

"""
calcF(Δ, kgrid, fdlr)

calculate the F function in τ-k representation 

# Arguments:

- Δ: gap function in (ωn, k) representation
- kgrid: momentum grid
- fdlr::DLRGrid: DLR Grid that contains the imaginary-time grid
"""
function calcF(Δ0, Δ, fdlr, k::CompositeGrid)
    Δ = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=2)
    F = zeros(ComplexF64, (k.Np * k.order, fdlr.size))
    for (ki, k) in enumerate(k.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
        end
    end
    F = DLR.matfreq2tau(:fermi, F, fdlr, fdlr.τ, axis=2)
    return real(F)
end

function dH1(k, p, τ)
    g = e0^2
    gh = sqrt(g)
    if abs(k - p) > 1.0e-12
        return -2π * gh^3 * (log((abs(k - p) + gh)) - log(abs(k - p))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
        # return -2π * gh^3 * (log((abs(k - p) + gh))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
    else
        return 0.0
end
end

function bare(k, p)
    if abs(k - p) > 1.0e-12
        return 4π * e0^2 * (log(k + p) - log(abs(k - p)))
        # return 4π * e0^2 * log(k + p)         
    else
        return 0.0
end
end


"""
interpolate F from a one-dimensional array of k grid to a two-diemnsional array of k and q grids

# Arguments
- F : one-dimensional array in k grid
- fdlr : dlrGrid object
- kgrid : CompositeGrid for k
- qgrids : vector of CompositeGrid, each element is a q grid for a given k grid
"""
function interpolateF(F, fdlr, kgrid, qgrids)
    @assert size(F)[2] == fdlr.size # check the τ dimension
    @assert length(qgrids) == kgrid.Np * kgrid.order

    FF = zeros(Float64, (qgrids[1].Np * qgrids[1].order, kgrid.Np * kgrid.order, fdlr.size))

    for τi in 1:fdlr.size
        for (ki, k) in enumerate(kgrid.grid)
            kpidx = 1 # panel index of the kgrid
            fx = F[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order), τi] # all F in the same kpidx-th K panel
            x = kgrid.grid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
            w = kgrid.wgrid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
            for (qi, q) in enumerate(qgrids[ki].grid)
                # for a given q, one needs to find the k panel to do interpolation
                if q > kgrid.panel[kpidx + 1]
                    # if q is too large, move k panel to the next
                    kpidx += 1
                    fx = F[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order), τi] # all F in the same kpidx-th K panel
                    x = kgrid.grid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
                    w = kgrid.wgrid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
                    @assert kpidx <= kgrid.Np
                end
                # extract all x in the kpidx-th k panel
                FF[qi, ki, τi] = barycheb(kgrid.order, q, fx, w, x) # the interpolation is independent with the panel length

            end
        end
    end

    return FF
end

# function interpolateF2(F, fdlr, kgrid, qgrids)
#     @assert size(F)[2] == fdlr.size # check the τ dimension
#     @assert length(qgrids) == kgrid.Np * kgrid.order

#     FF = zeros(Float64, (qgrids[1].Np * qgrids[1].order, kgrid.Np * kgrid.order, fdlr.size))

#     for (ki, k) in enumerate(kgrid.grid)
#         kpidx = 1 # panel index of the kgrid
#         fx = F[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order), :] # all F in the same kpidx-th K panel
#         x = kgrid.grid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
#         w = kgrid.wgrid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
#         # println("kfx: $kpidx: $(kgrid.x)")
#         # println("kfx: $kpidx: $fx")
#         for (qi, q) in enumerate(qgrids[ki].grid)
#             # for a given q, one needs to find the k panel to do interpolation
#             if q > kgrid.panel[kpidx + 1]
#                 # if q is too large, move k panel to the next
#                 kpidx += 1
#                 fx = F[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order), :] # all F in the same kpidx-th K panel
#                 x = kgrid.grid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
#                 w = kgrid.wgrid[idx(kpidx, 1, kgrid.order):idx(kpidx, kgrid.order, kgrid.order)]
#                 # println("$kpidx: $fx")
#                 @assert kpidx <= kgrid.Np
#             end
#             # extract all x in the kpidx-th k panel
#             FF[qi, ki, :] .= barycheb2(kgrid.order, q, fx, w, x) # the interpolation is independent with the panel length

#             # if qi < 10
#             #     println(FF[qi, ki, τi], " at ", q)
#             # else
#             #     exit(0)
#             # end
#         end
#         # FF[:, ki, τi] = interpolate(F[:, τi], kgrid, qgrids[ki].grid)
#     end

#     return FF
# end

"""
Calculate Δ function in k grid

# Arguments
- F : one-dimensional array in k grid
- fdlr : dlrGrid object
- kgrid : CompositeGrid for k
- qgrids : vector of CompositeGrid, each element is a q grid for a given k grid
"""
function calcΔ(F, fdlr, kgrid, qgrids)
    
    @time begin
        FF = interpolateF(F, fdlr, kgrid, qgrids)
    end
    
    # p = plot(kgrid.grid, F[:, 10])
    # p = plot!(p, kgrid.grid, FF[:, 1, 10])
    # display(p)
    # readline()
    
    Δ0 = zeros(Float64, kgrid.Np * kgrid.order)
    Δ = zeros(Float64, (kgrid.Np * kgrid.order, fdlr.size))
    
    for (τi, τ) in enumerate(fdlr.τ)
        for (ki, k) in enumerate(kgrid.grid)
            qgrid = qgrids[ki]
            for (qi, q) in enumerate(qgrid.grid)
                wqgrid = qgrid.wgrid
                Δ[ki, τi] += dH1(k, q, τ) * FF[qi, ki, τi] * wqgrid[qi]

                if τi == 1 
                    Δ0[ki] += bare(k, q) * FF[qi, ki, 1] * wqgrid[qi]
                end

            end
        end
    end
    return Δ0, Δ 
end

if abspath(PROGRAM_FILE) == @__FILE__
    # KGrid()
    fdlr = DLR.DLRGrid(:fermi, 10EF, β, 1e-10) 
                
    ########## non-uniform kgrid #############
    # kgrid, wkgrid = KGrid(64, 10)
    # kgrid_double, wkgrid_double = KGrid(32, 10)
    Nk = 8
    order = 16
    kpanel = KPanel(Nk)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    # qgrids = [CompositeGrid(QPanel(Nk, k), order, :cheb) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    qgrids = [CompositeGrid(QPanel(Nk, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    
    # kgrid_double = CompositeGrid(kpanel, 2 * order, :cheb)
    kgrid_double = CompositeGrid(kpanel, 2 * order, :cheb)
    # qgrids_double = [CompositeGrid(QPanel(Nk, k), 2 * order, :cheb) for k in kgrid_double.grid] # qgrid for each k in kgrid.grid
    qgrids_double = [CompositeGrid(QPanel(Nk, k), 2 * order, :gaussian) for k in kgrid_double.grid] # qgrid for each k in kgrid.grid
    
    Δ = zeros(Float64, (length(kgrid.grid), fdlr.size))
    Δ0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    F = calcF(Δ0, Δ, fdlr, kgrid)
    # filename = "./bare.dat"
    # open(filename, "w") do io
    #     for r in 1:length(kgrid)
    #         # @printf(io, "%32.17g  %32.17g %32.17g %32.17g\n",kgrid.grid[r] , bare(2.0, kgrid[r]) * F[1, r] * wkgrid[r], bare(2.0, kgrid[r]) * wkgrid[r], F[1, r] * wkgrid[r])
    #     end
    # end
    
    println("sumF: $(sum(F)), maximum: $(maximum(abs.(F)))")
    Δ0, Δ = calcΔ(F, fdlr, kgrid, qgrids)
    Δ_freq = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=2)
    
    Δ_2 = zeros(Float64, (length(kgrid_double.grid), fdlr.size))
    Δ0_2 = zeros(Float64, length(kgrid_double.grid)) .+ 1.0
    F = calcF(Δ0_2, Δ_2, fdlr, kgrid_double)
    println(sum(F))
    Δ0_double, Δ_double = calcΔ(F, fdlr, kgrid_double, qgrids_double)
    
    Δ0_fine = interpolate(Δ0_double, kgrid_double, kgrid.grid)
    println("Max Err: ", maximum(abs.(Δ0 - Δ0_fine)))
    # println("Max Err: ", maximum(abs.(Δ0 - Δ0_double)))
    # Δ_freq = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=1)
    # maxvl, maxindex= findmax(reshape(abs.(Δ-Δ_double), fdlr.size*length(kgrid)))
    # println(maxindex)
    # index1 = maxindex ÷
    maxindex = 1
    for r in 1:length(kgrid.grid)
        if kgrid.grid[r] < 2.0
            global maxindex = r
        end
    end
    # Δ0_fine = interpolate(Δ0_double, kgrid_double, kgrid.grid)
    # println(abs.(Δ0 - Δ0_fine)[maxindex], ",", kgrid.grid[maxindex])
    # println(abs.(Δ0 - Δ0_double)[maxindex], ",", kgrid.grid[maxindex])
    # println(abs.(Δ - Δ_double)[maxindex], ",", kgrid[maxindex])
    filename = "./test.dat"
    println(fdlr.n, fdlr.n[fdlr.size ÷ 2 + 1])
    open(filename, "w") do io
        for r in 1:length(kgrid.grid)
            @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid.grid[r] ,Δ0[r] ,real(Δ_freq[r, fdlr.size ÷ 2 + 1]))
        end
    end
    # filename = "./.dat"    
    # open(filename, "w") do io
    #     for r in 1:length(kgrid)
    #         @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid[r] ,Δ0[r] ,real(Δ_freq[fdlr.size÷2+1,r]))
    #     end
    # end
    end
