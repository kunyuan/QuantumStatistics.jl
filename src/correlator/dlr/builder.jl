include("chebyshev.jl")
using Printf

struct CompositeChebyshevGrid
    degree::Int # Chebyshev degree
    x::Vector{Float64} # Chebyshev nodes
    w::Vector{Float64} # Chebyshev node weights

    np::Int # number of panel
    panel::Vector{Float64}

    ngrid::Int # size of the grid = (np-1)*degree
    grid::Vector{Float64}  # fine grid
    function CompositeChebyshevGrid(degree, panel)
        # fill each panel with N order Chebyshev nodes
        np = length(panel) # number of panels break points
        xc, wc = barychebinit(degree)
        fineGrid = zeros(Float64, (np - 1) * degree) # np break points have np-1 panels
        for i in 1:np - 1
            a, b = panel[i], panel[i + 1]
            fineGrid[(i - 1) * degree + 1:i * degree] = a .+ (b - a) .* (xc .+ 1.0) ./ 2
        end
        return new(degree, xc, wc, np, panel, (np - 1) * degree, fineGrid)
    end
end

"""
kernalDiscretization(type, Nτ, Nω, Λ, rtol)

Discretize kernel K(tau,omega) on composite Chebyshev fine grids for τ and ω. 
Generate a panels and grids for τ and ω.

#Arguments:
- `type`: :fermi, :bose or :corr
- `Dτ`: Chebyshev degree in each τ panel
- `Dω`: Chebyshev degree in each ω panel
- `Λ`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance relative error

#Returns
- `τPanel`: panel break points for τ, exponentially get dense near 0⁺ and 1⁻
- `ωPanel`: panel break points for ω, get exponentially dense near 0⁻ and 0⁺
- `τGrid`: tau fine grid points on (0,1)
- `ωGrid`: omega fine grid points on (-Λ, Λ)
- `kernel`: K(tau,omega) on fine grid 
- `err`: Error of composite Chebyshev interpolant of K(tau,omega). err(1) is ~= max relative L^inf error in tau over all omega in fine grid. err(2) is ~= max L^inf error in omega over all tau in fine grid.
"""
function kernalDiscretization(type, Dτ, Dω, Λ, rtol)
    npt = Int(ceil(log(Λ) / log(2.0))) - 2 # subintervals on [0,1/2] in tau space (# subintervals on [0,1] is 2*npt)
    npo = Int(ceil(log(Λ) / log(2.0))) # subintervals on [0,lambda] in omega space (subintervals on [-lambda,lambda] is 2*npo)


    if type == :corr
        ############# Tau discretization ##############
        # Panel break points for the imaginary time ∈ (0, 1)
        # get exponentially dense near 0⁺ 
        pbpt = zeros(Float64, npt + 1)
        pbpt[1] = 0.0
        for i in 1:npt
            pbpt[i + 1] = 0.5 / 2^(npt - i)
        end

        # Panel break points for the real frequency ∈ [0, Λ]
        # get exponentially dense near 0⁺
        pbpo = zeros(Float64, npo + 1)
        pbpo[1] = 0.0
        for i in 1:npo
            pbpo[i + 1] = Λ / 2^(npo - i)
        end
    else
        ############# Tau discretization ##############
        # Panel break points for the imaginary time ∈ (0, 1)
        # get exponentially dense near 0⁺ and 1⁻ 
        pbpt = zeros(Float64, 2npt + 1)
        pbpt[1] = 0.0
        for i in 1:npt
            pbpt[i + 1] = 1.0 / 2^(npt - i + 1)
        end
        pbpt[npt + 2:2npt + 1] = 1 .- pbpt[npt:-1:1]

        ############ ω discretization ##################
        # Panel break points for the real frequency ∈ [-Λ, Λ]
        # get exponentially dense near 0⁻ and 0⁺
        pbpo = zeros(Float64, 2npo + 1)
        pbpo[npo + 1] = 0.0
        for i in 1:npo
            pbpo[npo + i + 1] = Λ / 2^(npo - i)
        end
        pbpo[1:npo] = -pbpo[2npo + 1:-1:npo + 2]
    end

    # Grid points
    τ = CompositeChebyshevGrid(Dτ, pbpt)
    ω = CompositeChebyshevGrid(Dτ, pbpo)

    # kernel = Spectral.kernelT(type, τ.grid, ω.grid, 1.0) # β = 1.0
    kernel = preciseKernel(type, τ, ω)

    println("=====  Kernel Discretization =====")
    println("fine grid points for τ     = ", τ.ngrid)
    println("fine grid points for ω     = ", ω.ngrid)

    return τ, ω, kernel
end

function preciseKernel(type, τ, ω)
    # Assume τ.grid is particle-hole symmetric!!!
    @assert (τ.np - 1) * τ.degree == τ.ngrid
    kernel = zeros(Float64, (τ.ngrid, ω.ngrid))

    if type==:fermi
        #K(τ, ω)=K(β-τ, -ω) for τ>0 
        @assert isodd(τ.np) #symmetrization is only possible for odd τ panels
        halfτ = ((τ.np - 1) ÷ 2) * τ.degree
        kernel[1:halfτ, :] = Spectral.kernelT(type, τ.grid[1:halfτ], ω.grid, 1.0)
        kernel[end:-1:(halfτ + 1), :] = Spectral.kernelT(type, τ.grid[1:halfτ], ω.grid[end:-1:1], 1.0)
    elseif type==:corr
        kernel = Spectral.kernelT(type, τ.grid, ω.grid, 1.0)
        # kernel[end:-1:(halfτ + 1), :] = Spectral.kernelT(type, τ.grid[1:halfτ], ω.grid, 1.0)
        # @assert all(kernel[1, :] ≈ kernel[end, :]) 
    else
        error("$type isn't implemented!")
    end
    return kernel
end

function testInterpolation(type, τ, ω, kernel)
    ############# test interpolation accuracy in τ #######
    τ2 = CompositeChebyshevGrid(τ.degree * 2, τ.panel)
    kernel2 = preciseKernel(type, τ2, ω)
    err = 0.0
    for ωi in 1:length(ω.grid)
        tmp = 0.0
        for i in 1:τ2.np - 1
            for k in 1:τ2.degree
                τidx = (i - 1) * τ2.degree + k
                kaccu = kernel2[τidx, ωi]
                kinterp = barycheb(τ.degree, τ2.x[k], kernel[(i - 1) * τ.degree + 1:i * τ.degree, ωi], τ.w, τ.x)
                tmp = max(tmp, abs(kaccu - kinterp))
            end
        end 
        err = max(err, tmp / maximum(kernel[:,ωi]))
    end
    println("Max relative L∞ error of kernel discretization in τ = ", err)

    ω2 = CompositeChebyshevGrid(ω.degree * 2, ω.panel)
    kernel2 = preciseKernel(type, τ, ω2)
    err = 0.0
    for τi in 1:length(τ.grid)
        tmp = 0.0
        for i in 1:ω2.np - 1
            for k in 1:ω2.degree
                idx = (i - 1) * ω2.degree + k
                kaccu = kernel2[τi, idx]
                kinterp = barycheb(ω.degree, ω2.x[k], kernel[τi, (i - 1) * ω.degree + 1:i * ω.degree], ω.w, ω.x)
                tmp = max(tmp, abs(kaccu - kinterp))
            end
        end 
        err = max(err, tmp / maximum(kernel[τi, :]))
    end
    println("Max relative L∞ error of kernel discretization in ω = ", err)
end

"""
function dlr(type, Λ, rtol)
    Construct discrete Lehmann representation

#Arguments:
- `type`: type of kernel, :fermi, :boson
- `Λ`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance absolute error
"""
function dlr(type, Λ, rtol)
    Λ = Float64(Λ)
    @assert 0.0 < rtol < 1.0
    degree = 24 # number of Chebyshev nodes in each panel
    τ, ω, kernel = kernalDiscretization(type, degree, degree, Λ, rtol)
    testInterpolation(type, τ, ω, kernel)
    τGrid, ωGrid = τ.grid, ω.grid
    Nτ, Nω = length(τGrid), length(ωGrid)
    @assert size(kernel)==(Nτ, Nω)
    println(τ.grid[end], ", ", τ.panel[end])
    println(ω.grid[end], ", ", ω.panel[end])
    ################# find the rank ##############################
    """
    For a given index k, decompose R=[R11, R12; 0, R22] where R11 is a k×k matrix. 
    If R11 is well-conditioned, then 
    σᵢ(R11) ≤ σᵢ(kernel) for 1≤i≤k, and
    σⱼ(kernel) ≤ σⱼ₋ₖ(R22) for k+1≤j≤N
    See Page 487 of the book: Golub, G.H. and Van Loan, C.F., 2013. Matrix computations. 4th. Johns Hopkins.
    Thus, the effective rank is defined as the minimal k that satisfy rtol≤ σ₁(R22)/σ₁(kernel)
    """
    u, σ, v = svd(kernel)
    rank, err = 1, 0.0
    for (si, s) in enumerate(σ)
        # println(si, " => ", s / σ[1])
        if s / σ[1] < rtol
            rank = si - 1
            err = s[1] / σ[1]
    break
        end
    end
    println("Kernel ϵ-rank = ", rank, ", rtol ≈ ", err)

    Q, R, p = qr(kernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    # size(R) == (Nτ, Nω) if Nω>Nτ
    # or size(R) == (Nω, Nω) if Nω<Nτ

    for idx in rank:min(Nτ, Nω) 
        if Nω>Nτ
            R22 = R[idx:Nτ, idx:Nω]
        else
            R22 = R[idx:Nω, idx:Nω]
        end
        u2, s2, v2 = svd(R22)
        # println(idx, " => ", s2[1] / σ[1])
        if s2[1] / σ[1] < rtol
            rank = idx
            err = s2[1] / σ[1]
    break
        end
    end
    println("DLR rank      = ", rank,  ", rtol ≈ ", err)

    # @assert err ≈ 4.58983288255442e-13

    ###########  dlr grid for ω  ###################
    println("Calculating τ grid ...")
    ωGridDLR = sort(ωGrid[p[1:rank]])

    ###########  dlr grid for τ  ###################
    τkernel = zeros(Float64, (rank, Nτ))
    for τi in 1:Nτ
        for r in 1:rank
            τkernel[r, τi] = kernel[τi, p[r]]
        end
    end
    τqr = qr(τkernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    τGridDLR = sort(τGrid[τqr.p[1:rank]])
    
    ##########  dlr grid for ωn  ###################
    println("Calculating ωn grid ...")

    if type==:corr
        Nωn = Int(ceil(Λ)) * 2 # expect Nω ~ para.Λ/2π, drop 2π on the safe side
        ωnkernel = zeros(Float64, (rank, Nωn + 1))
        ωnGrid = [w for w in 0:Nωn] # fermionic Matsubara frequency ωn=(2n+1)π
        # ωnkernel = zeros(Complex{Float64}, (rank, 2Nωn + 1))
        # ωnGrid = [w for w in -Nωn:Nωn] # fermionic Matsubara frequency ωn=(2n+1)π
    else
        Nωn = Int(ceil(Λ)) * 2 # expect Nω ~ para.Λ/2π, drop 2π on the safe side
        ωnkernel = zeros(Complex{Float64}, (rank, 2Nωn + 1))
        ωnGrid = [w for w in -Nωn:Nωn] # fermionic Matsubara frequency ωn=(2n+1)π
    end

    for (ni, n) in enumerate(ωnGrid)
        for r in 1:rank
            ωnkernel[r, ni] = Spectral.kernelΩ(type, n, ωGridDLR[r])
        end
    end
    nqr = qr(ωnkernel, Val(true)) # julia qr has a strange, Val(true) will do a pivot QR
    nGridDLR = sort(ωnGrid[nqr.p[1:rank]])

    ########### output  ############################
    @printf("%5s  %32s  %32s  %8s\n", "index", "real freq", "tau", "ωn")
        for r in 1:rank
        @printf("%5i  %32.17g  %32.17g  %8i\n", r, ωGridDLR[r], τGridDLR[r], nGridDLR[r])
    end

    dlr = Dict([(:ω, ωGridDLR), (:τ, τGridDLR), (:ωn, nGridDLR)])
    return dlr
end