using StaticArrays:similar, maximum
using QuantumStatistics
using LegendrePolynomials
using Printf
# using Gaston
using Plots

srcdir = "."
rundir = isempty(ARGS) ? "." : (pwd()*"/"*ARGS[1])

if !(@isdefined β)
    include(rundir*"/parameter.jl")
    using .parameter
end

include("grid.jl")


function KGrid_bose(Order, Nk)
    maxK = 10.0
    minK = 0.001
    # Nk = 10
    # Order = 8 # Legendre polynomial order
    panel = Grid.boseKUL(0.5 * kF, maxK, minK, Nk, 1).grid
    panel[1] = 0.0  # the kgrid start with 0.0
    println("Panels Num: ", length(panel))

    # n: polynomial order
    x, w = gausslegendre(Order)
    println("Guassian quadrature points : ", x)
    println("Guassian quadrature weights: ", w)

    println("KGrid Num: ", (length(panel) - 1) * Order)

    kgrid = zeros(Float64, (length(panel) - 1) * Order)
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
    # kgrid[length(kgrid)]=maxK
    return kgrid, wkgrid
end


function build_int(k, q, kpanel_bose, int_order)
    #println(kbose_grid.panel)
    if abs(k-q) <  1e-12
        kmin = 1e-12
    else
        kmin = abs(k-q)
    end

    int_panel = Float64[]

    push!(int_panel, kmin)
    for (pi, p) in enumerate(kpanel_bose)
        if (p > kmin && p < k + q)
            push!(int_panel, p)
        end
    end
    push!(int_panel, k + q)
    #println("int_panel:$(int_panel)")
    grid_int = CompositeGrid(int_panel,int_order, :gaussian)
    #println("kmin= $(kmin), k=$(k), q=$(q), int_grid:$(grid_int.grid)")
    return grid_int
end

function legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, int_order)
    kernal_bare = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid)))
    kernal = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid), bdlr.size))
    for (ki, k) in enumerate(kgrid.grid)
        for (pi, p) in enumerate(qgrids[ki].grid)
            for (ni, n) in enumerate(bdlr.n)
                if abs(k - p) > 1.0e-12
                    grid_int = build_int(k, p ,kpanel_bose, int_order)
                    kernal_bare[ki,pi], kernal[ki,pi,ni] = Composite_int(k, p, n, grid_int)
                    @assert isfinite(kernal[ki,pi,ni]) "fail kernal at $ki,$pi,$ni, with $(kernal[ki,pi,ni])"
                else
                    kernal_bare[ki,pi] = 0
                    kernal[ki,pi,ni] = 0
                end
            end
        end
    end
 
    return kernal_bare,  kernal
end

function Composite_int(k, p, n, grid_int)
    sum = 0
    sum_bare = 0
    g = e0^2
    for (qi, q) in enumerate(grid_int.grid)
        legendre_x = (k^2 + p^2 - q^2)/2/k/p
        if(abs(abs(legendre_x)-1)<1e-12)
            legendre_x = sign(legendre_x)*1
        end
        wq = grid_int.wgrid[qi]
        sum += q*Pl(legendre_x, channel)*RPA(q, n) * wq
        sum_bare += Pl(legendre_x, channel)*4*π*g/q * wq
    end
    return sum_bare, sum
end

"""
calcF(Δ, kgrid, fdlr)

calculate the F function in τ-k representation 

# Arguments:

- Δ: gap function in (ωn, k) representation
- kgrid: momentum grid
- fdlr::DLRGrid: DLR Grid that contains the imaginary-time grid
"""
function calcF(Δ0, Δ, fdlr, k::CompositeGrid)
    Δ = DLR.tau2matfreq(:acorr, Δ, fdlr, fdlr.n, axis=2)
    #F = zeros(ComplexF64, (length(k.grid), fdlr.size))
    F = zeros(ComplexF64, (length(k.grid), fdlr.size))

    for (ki, k) in enumerate(k.grid)
        ω = k^2 / (2me) - EF
        for (ni, n) in enumerate(fdlr.n)
            np = n # matsubara freqeuncy index for the upper G: (2np+1)π/β
            nn = -n - 1 # matsubara freqeuncy for the upper G: (2nn+1)π/β = -(2np+1)π/β
            F[ki, ni] = (Δ[ki, ni] + Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
            #F[ki, ni] = (Δ[ki, ni]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
      
        end

    end
    
    #println("F_imag=$(F_max)")
    F = DLR.matfreq2tau(:acorr, F, fdlr, fdlr.τ, axis=2)
    # gg_τ = GG_τ(kgrid, fdlr.τ)
    
    # for (ki, k) in enumerate(k.grid)
    #     for (ni, n) in enumerate(fdlr.τ)
    #         F[ki, ni] = F[ki, ni] + Δ0[ki] * gg_τ[ki, ni]
    #     end
    # end
   
    return  real(F)
end

function dH1(k, p, τ)
    g = e0^2
    gh = sqrt(g)
    if abs(k - p) > 1.0e-12
        return -2π * gh^3 * log((abs(k - p) + gh) / (abs(k - p))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
        # return -2π * gh^3 * (log((abs(k - p) + gh))) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
    else
        return 0.0
end
end

function dH1_freq(kgrid, qgrids, bdlr, fdlr)
    g = e0^2
    kernal_bare = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid)))
    kernal = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid), bdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        for (pi, p) in enumerate(qgrids[ki].grid)
            if abs(k - p) > 1.0e-12
                kernal_bare[ki,pi] = bare(k, p)
            else
                kernal_bare[ki,pi] = 0
            end
            for (ni,n) in enumerate(bdlr.n)
                if abs(k - p) > 1.0e-12
                    ω_n = 2*π*n/β
                    kernal[ki,pi,ni] = -2*π*g^2/(ω_n^2+g)*( log((k+p)^2/(k-p)^2) - log(((k + p)^2 + ω_n^2 + g) / ((k - p)^2 + ω_n^2 + g)) )
                else
                    kernal[ki,pi,ni] = 0
                end
            end
        end
    end

    #result_τ = DLR.matfreq2tau(:corr, kernal, bdlr, fdlr.τ, axis=3)
    #println(maximum(abs.(imag.(result_τ))))
    #return real.(result_τ)

    return kernal_bare, kernal
end

function dH1_bose(q, n)
    g = e0^2
    kernal = 0.0
    if abs(q) > 1.0e-12
        ω_n = 2*π*n/β
        kernal = -4*π*g^2/(q^2+ω_n^2+g)/q^2
    else
        kernal = 0
                                
    end

   return kernal
end




function dH1_tau(kgrid, qgrids, fdlr)
    g = e0^2
    kernal = zeros(Float64, (length(kgrid.grid), length(qgrids[1].grid), fdlr.size))

    for (ki, k) in enumerate(kgrid.grid)
        for (pi, p) in enumerate(qgrids[ki].grid)
            for (τi, τ) in enumerate(fdlr.τ)
                if abs(k - p) > 1.0e-12
                   
                    kernal[ki,pi,τi] = dH1(k, p, τ)
                else
                    kernal[ki,pi,τi] = 0
                end
            end
        end
    end

    # result_τ = DLR.matfreq2tau(:fermi, kernal, fdlr, fdlr.τ, axis=3)
    # println(maximum(abs.(imag.(result_τ))))
    return kernal
end

function RPA(q, n)
    g = e0^2
    kernal = 0.0
    if abs(q) > 1.0e-12
        x = q/2/kF
        ω_n = 2*π*n/β
        y = me*ω_n/q/kF
        #Π = me*kF/2/π^2*(1 + (1 -x^2 + y^2)*log(((1+x)^2+y^2)/((1-x)^2+y^2))/4/x - y*atan( 2*y/(y^2+x^2-1) ))
        Π = me*kF/2/π^2*(1 + (1 -x^2 + y^2)*log1p(4*x/((1-x)^2+y^2))/4/x - y*(atan( 2*y/(y^2+x^2-1) ) ))
        kernal = -4*π*g/q^2* Π/( q^2/4/π/g  + Π )
        #kernal = Π2
        #println("test_RPA: $(Π)")
        #println("test_RPA: $(Π2)")
    else
        kernal = 0
        
    end

    return kernal
end



function bare(k, p)
    if abs(k - p) > 1.0e-12
        return 4π * e0^2 * log((k + p)/(abs(k - p)))
        # return 4π * e0^2 * log(k + p)         
    else
        return 0.0
    end
    #return 4π * e0^2 * log((k + p)/(abs(k - p)))
end


"""
Calculate Δ function in k grid

# Arguments
- F : one-dimensional array in k grid
- fdlr : dlrGrid object
- kgrid : CompositeGrid for k
- qgrids : vector of CompositeGrid, each element is a q grid for a given k grid
"""
function calcΔ(F,  kernal, kernal_bare, fdlr, kgrid, qgrids)
    
    Δ0 = zeros(Float64, length(kgrid.grid))
    Δ = zeros(Float64, (length(kgrid.grid), fdlr.size))
    order = kgrid.order

    F_ins = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    F_ins = DLR.dlr2tau(:acorr, F_ins, fdlr, [1.0e-12,] , axis=2)[:,1]

    #F_ins2 = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    #F_ins2 = DLR.dlr2tau(:acorr, F_ins2, fdlr, [β-1.0e-12,] , axis=2)[:,1]
    #println("F_ins[0] = $((F_ins[1])), F_ins[β]=$((F_ins2[1])) ")
    for (ki, k) in enumerate(kgrid.grid)
        
        kpidx = 1 # panel index of the kgrid
        head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
        x = @view kgrid.grid[head:tail]
        w = @view kgrid.wgrid[head:tail]

        for (qi, q) in enumerate(qgrids[ki].grid)
            
            if q > kgrid.panel[kpidx + 1]
                # if q is larger than the end of the current panel, move k panel to the next panel
                while q > kgrid.panel[kpidx + 1]
                    kpidx += 1
                end
                head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
                x = @view kgrid.grid[head:tail]
                w = @view kgrid.wgrid[head:tail]
                @assert kpidx <= kgrid.Np
            end

            
            for (τi, τ) in enumerate(fdlr.τ)

                fx = @view F[head:tail, τi] # all F in the same kpidx-th K panel
                FF = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length

                wq = qgrids[ki].wgrid[qi]
                # Δ[ki, τi] += dH1(k, q, τ) * FF * wq
                Δ[ki, τi] += kernal[ki ,qi ,τi] * FF * wq
                @assert isfinite(Δ[ki, τi]) "fail Δ at $ki, $τi with $(Δ[ki, τi]), $FF\n $q for $fx\n $x \n $w\n $q < $(kgrid.panel[kpidx + 1])"

                if τi == 1
                    fx_ins = @view F_ins[head:tail] # all F in the same kpidx-th K panel
                    FF = barycheb(order, q, fx_ins, w, x) # the interpolation is independent with the panel length
                    Δ0[ki] += bare(k, q) * FF * wq
                    Δ0[ki] += kernal_bare[ki, qi] * FF * wq                    
                    @assert isfinite(Δ0[ki]) "fail Δ0 at $ki with $(Δ0[ki])"
                end

            end
        end
    end
    
    return Δ0, Δ 
end

function GG_τ(kgrid, tau)
    GG = zeros(Float64, (length(kgrid.grid), length(tau)))
    for (ki,k) in enumerate(kgrid.grid)
        ω = (k^2 / (2me) - EF)
        for (τi, τ) in enumerate(tau)
            GG[ki,τi] = (exp(-ω*τ)-exp(-ω*(β-τ)))/(1+exp(-ω*β))/2.0/ω
            
            #GG[ki, τi] = exp(-x*y)/2.0/cosh(x)
        end
    end
    return GG    
end



function testGrid(kgrid, qgrids, qgrids2, F)
    
    # plotlyjs() # allow interactive plot
    # pyplot() # allow interactive plot
    F = F[:, 10]
    ki = findall(x -> abs(x - kF * 1.2) < 1.0e-2, kgrid.grid)
    println("ki: $ki")
    ki = ki[1]
    qgrid = qgrids[ki]
    qgrid2 = qgrids2[ki]
    
    integrand = zeros(Float64, length(qgrid.grid))
    for (qi, q) in enumerate(qgrid.grid)
        FF = interpolate(F, kgrid, qgrid.grid)
        integrand[qi] = bare(kgrid.grid[ki], q) * FF[qi]
    end
    
    integrand2 = zeros(Float64, length(qgrid2.grid))
    for (qi, q) in enumerate(qgrid2.grid)
        FF = interpolate(F, kgrid, qgrid2.grid)
        integrand2[qi] = bare(kgrid.grid[ki], q) * FF[qi]
    end
    
    p = plot(qgrid.grid ./ kF, integrand, curveconf="w p", Axes(xrange=(1.1, 1.4)))
    plot!(qgrid2.grid ./ kF, integrand2, curveconf="w p")
    # xlims!((1.1, 1.4))
    display(p)
    readline()
end

if abspath(PROGRAM_FILE) == @__FILE__
    
    fdlr = DLR.DLRGrid(:acorr, 10EF, β, 1e-10) 
    bdlr = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 
    ########## non-uniform kgrid #############
    # Nk = 16
    # order = 8
    # maxK = 10.0 * kF
    # minK = 0.00001 * kF
    
    kpanel = KPanel(Nk, kF, maxK, minK)
    kpanel_bose = KPanel(2Nk, 2*kF, 2.1*maxK, minK/100.0)
    kgrid = CompositeGrid(kpanel, order, :cheb)
    #kgrid_bose = CompositeGrid(kpanel_bose, order, :gaussian)
    qgrids = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid

    #println("kgrid: $(kgrid.grid)")
    #println("kgrid_bose: $(kgrid_bose.grid)")
    println("max qgrid number: ", maximum([length(q.grid) for q in qgrids]))
    
    qgrid2s = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2order, :gaussian) for k in kgrid.grid] # qgrid for each k in kgrid.grid

    q_test = 1e-8
    test_RPA = RPA(q_test , 1) / q_test^2
    println("$(test_RPA)")
    kernal_bare_int, kernal_int = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, 2*order)
    
    #kernal_int_double = legendre_dc(bdlr, kgrid, qgrids, kpanel_bose, 4*order)
    kernal_bare, kernal = dH1_freq(kgrid, qgrids, bdlr, fdlr)
    println("legendre_err: $(maximum(abs.(kernal_bare .- kernal_bare_int)))  , $(maximum(abs.(kernal .- kernal_int)))")
    #println("legendre_err: $(maximum(abs.(kernal_int_double .- kernal_int)))")
    println("$(kgrid.grid[1]),$((bdlr.n)[bdlr.size÷2+1])")
    #kernal = dH1_tau(kgrid, qgrids, fdlr)
    # xMin=0.99999
    # xMax=1.00001
    # yMin=-0.003
    # yMax=0.003
    # p = plot(qgrids[1].grid, kernal[1, :, bdlr.size÷2+1]) #, xlim=(xMin,xMax), ylim=(yMin, yMax))
    # p = plot!(p, qgrids[1].grid, kernal_int[1, :, bdlr.size÷2+1])
    # display(p)
    # readline()
    

    kgrid_double = CompositeGrid(kpanel, 2 * order, :cheb)
    # kgrid_double = CompositeGrid(kpanel, order, :cheb)
    qgrids_double = [CompositeGrid(QPanel(Nk, kF, maxK, minK, k), 2order, :gaussian) for k in kgrid_double.grid] # qgrid for each k in kgrid.grid

    #kernal_double = dH1_freq(kgrid, qgrids, fdlr)
    kernal_double = dH1_tau(kgrid_double, qgrids_double, fdlr)

   

    
    Δ = zeros(Float64, (length(kgrid.grid), fdlr.size))
    Δ0 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    F = calcF(Δ0, Δ, fdlr, kgrid)
    gg_τ = GG_τ(kgrid, [1.0e-12,])
    F_ins = DLR.tau2dlr(:acorr, F, fdlr, axis=2)
    F_ins = DLR.dlr2tau(:acorr, F_ins, fdlr, [1.0e-12,] , axis=2)[:,1]
    # testGrid(kgrid, qgrids, qgrid2s, F)
    # println(size(F))
    # filename = "./bare.dat"
    # open(filename, "w") do io
    #     for r in 1:length(kgrid)
    #         # @printf(io, "%32.17g  %32.17g %32.17g %32.17g\n",kgrid.grid[r] , bare(2.0, kgrid[r]) * F[1, r] * wkgrid[r], bare(2.0, kgrid[r]) * wkgrid[r], F[1, r] * wkgrid[r])
    #     end
    # end
 
    #findall(x->x == maximum(abs.(abs.(gg_τ) - gg_τ_test)),abs.(abs.(gg_τ) - gg_τ_test))
  
    println("sumF: $(sum(F)), maximum: $(maximum(abs.(F)))")
    
    printstyled("Calculating Δ\n", color=:yellow)
    @time Δ0, Δ = calcΔ(F, kernal, fdlr ,kgrid, qgrids)
  
    fdlr2 = DLR.DLRGrid(:acorr, 100EF, β, 1e-10) 
    bdlr2 = DLR.DLRGrid(:corr, 100EF, β, 1e-10) 

    kernal_2 = dH1_tau(kgrid, qgrids, fdlr2)

    Δ_2 = zeros(Float64, (length(kgrid.grid), fdlr2.size))
    Δ0_2 = zeros(Float64, length(kgrid.grid)) .+ 1.0
    F_2 = calcF(Δ0_2, Δ_2, fdlr2, kgrid)
    @time Δ0_2, Δ_2 = calcΔ(F_2, kernal_2, fdlr2 ,kgrid, qgrids)

    Δ_freq = DLR.tau2matfreq(:acorr, Δ, fdlr, fdlr.n, axis=2)
    Δ_freq_2 = DLR.tau2matfreq(:acorr, Δ_2, fdlr2, fdlr.n, axis=2)
    err0 = maximum(abs.(Δ0_2-Δ0))
    err = maximum(abs.(Δ_freq_2-Δ_freq))
    ind=findmax(abs.(Δ0_2-Δ0))[2]
    println("err0: $(err0), err: $(err), mom: $(kgrid.grid[ind[1]])")
    # Δ_2 = zeros(Float64, (length(kgrid_double.grid), fdlr.size))
    # Δ0_2 = zeros(Float64, length(kgrid_double.grid)) .+ 1.0
    # F_2 = calcF(Δ0_2, Δ_2, fdlr, kgrid_double)
    # # println(sum(F_2))
    
    # F_fine = similar(F)
    # for τi in 1:fdlr.size
    #     F_fine[:, τi] = interpolate(F_2[:, τi], kgrid_double, kgrid.grid)
    # end
    # # p = plot(kgrid.grid, F[:, 10])
    # # p = plot!(p, kgrid_double.grid, F_2[:, 10])
    # # p = plot!(p, kgrid.grid, F_fine[:, 10])
    # # display(p)
    # # readline()
    # printstyled("Max Err for F interpolation: ", maximum(abs.(F - F_fine)), "\n", color=:red)
    
    # # println(Utility.red()
    
    # printstyled("Calculating Δ_double\n", color=:yellow)
    # @time Δ0_double, Δ_double = calcΔ(F_2, kernal_double, fdlr, kgrid_double, qgrids_double)
    
    # Δ0_fine = interpolate(Δ0_double, kgrid_double, kgrid.grid)
    # printstyled("Max Err for Δ0 interpolation: ", maximum(abs.(Δ0 - Δ0_fine)), "\n", color=:red)
    # # Δ_freq = DLR.tau2matfreq(:fermi, Δ, fdlr, fdlr.n, axis=1)
    # # maxvl, maxindex= findmax(reshape(abs.(Δ-Δ_double), fdlr.size*length(kgrid)))
    # # println(maxindex)
    # # index1 = maxindex ÷
    maxindex = 1
    for r in 1:length(kgrid.grid)
        if kgrid.grid[r] < 2.0
            global maxindex = r
        end
    end
    filename = "./test.dat"
    println(fdlr.n, fdlr.n[fdlr.size ÷ 2 + 1])
    open(filename, "w") do io
        for r in 1:length(kgrid.grid)
            @printf(io, "%32.17g  %32.17g  %32.17g %32.17g\n",kgrid.grid[r] ,Δ0[r], real(Δ_freq[r, fdlr.size ÷ 2 + 1]), Δ0[r]+real(Δ_freq[r, fdlr.size ÷ 2 + 1]))
        end
    end
    println("Max:",(maximum(abs.(Δ0[:])))/4/π^2,"\t", maximum(abs.(real(Δ_freq[:, fdlr.size ÷ 2 + 1])))/4/π^2,"\t", maximum(real( Δ0[:]+Δ_freq[:, fdlr.size ÷ 2 + 1])))
    # filename = "./.dat"    
    # open(filename, "w") do io
    #     for r in 1:length(kgrid)
    #         @printf(io, "%32.17g  %32.17g  %32.17g\n",kgrid[r] ,Δ0[r] ,real(Δ_freq[fdlr.size÷2+1,r]))
    #     end
    # end
    end
