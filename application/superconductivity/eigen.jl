using StaticArrays:similar, maximum
using QuantumStatistics
using Printf
# using Plots
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
Calculate Δ function in k grid

# Arguments
- F : one-dimensional array in k grid
- fdlr : dlrGrid object
- kgrid : CompositeGrid for k
- qgrids : vector of CompositeGrid, each element is a q grid for a given k grid
"""
function calcΔ(F, fdlr, kgrid, qgrids)
    
    @time begin
    
        Δ0 = zeros(Float64, kgrid.Np * kgrid.order)
        Δ = zeros(Float64, (kgrid.Np * kgrid.order, fdlr.size))
        order = kgrid.order
        
        for (τi, τ) in enumerate(fdlr.τ)
            for (ki, k) in enumerate(kgrid.grid)

                kpidx = 1 # panel index of the kgrid
                fx = F[idx(kpidx, 1, order):idx(kpidx, order, order), τi] # all F in the same kpidx-th K panel
                x = kgrid.grid[idx(kpidx, 1, order):idx(kpidx, order, order)]
                w = kgrid.wgrid[idx(kpidx, 1, order):idx(kpidx, order, order)]

                for (qi, q) in enumerate(qgrids[ki].grid)

                    if q > kgrid.panel[kpidx + 1]
                        # if q is too large, move k panel to the next
                        kpidx += 1
                        fx = F[idx(kpidx, 1, order):idx(kpidx, order, order), τi] # all F in the same kpidx-th K panel
                        x = kgrid.grid[idx(kpidx, 1, order):idx(kpidx, order, order)]
                        w = kgrid.wgrid[idx(kpidx, 1, order):idx(kpidx, order, order)]
                        @assert kpidx <= kgrid.Np
                    end

                    FF = barycheb(kgrid.order, q, fx, w, x) # the interpolation is independent with the panel length

                    wq = qgrids[ki].wgrid[qi]
                    Δ[ki, τi] += dH1(k, q, τ) * FF * wq

                    if τi == 1 
                        Δ0[ki] += bare(k, q) * FF * wq
                    end

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
    Nk = 16
    order = 8
    kpanel = KPanel(Nk)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    # qgrids = [CompositeGrid(QPanel(Nk, k), order, :cheb) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    qgrids = [CompositeGrid(QPanel(Nk, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid
    
    kgrid_double = CompositeGrid(kpanel, 2 * order, :cheb)
    # kgrid_double = CompositeGrid(kpanel, order, :cheb)
    # qgrids_double = [CompositeGrid(QPanel(Nk, k), 2 * order, :cheb) for k in kgrid_double.grid] # qgrid for each k in kgrid.grid
    qgrids_double = [CompositeGrid(QPanel(Nk, k), 2 * order, :gaussian) for k in kgrid_double.grid] # qgrid for each k in kgrid.grid
    
    Δ = zeros(Float64, (length(kgrid.grid), fdlr.size))
    Δ0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    F = calcF(Δ0, Δ, fdlr, kgrid)
    # println(size(F))
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
    F_2 = calcF(Δ0_2, Δ_2, fdlr, kgrid_double)
    # println(sum(F_2))

    F_fine = similar(F)
    for τi in 1:fdlr.size
        F_fine[:, τi] = interpolate(F_2[:, τi], kgrid_double, kgrid.grid)
    end
    # p = plot(kgrid.grid, F[:, 10])
    # p = plot!(p, kgrid_double.grid, F_2[:, 10])
    # p = plot!(p, kgrid.grid, F_fine[:, 10])
    # display(p)
    # readline()

    println("Max Err for F interpolation: ", maximum(abs.(F - F_fine)))

    Δ0_double, Δ_double = calcΔ(F_2, fdlr, kgrid_double, qgrids_double)
    
    Δ0_fine = interpolate(Δ0_double, kgrid_double, kgrid.grid)
    println("Max Err for Δ0: ", maximum(abs.(Δ0 - Δ0_fine)))
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
