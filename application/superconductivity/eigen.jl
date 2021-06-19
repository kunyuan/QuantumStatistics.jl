using StaticArrays:similar
using QuantumStatistics
include("parameter.jl")

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
            F[ni, ki] = (Δ[ni, ki]+Δ0[ki]) * Spectral.kernelFermiΩ(nn, ω, β) * Spectral.kernelFermiΩ(np, ω, β)
    end
end
    F = DLR.matfreq2tau(:fermi, F, fdlr, fdlr.τ, axis=1)
    return real(F)
end

function dH1(k, p, τ)
    g = 1.0
    gh = sqrt(g)
    if abs(k - p) > 1.0e-12
        return -2π * gh^3 * log((abs(k - p) + g) / abs(k - p)) * (exp(-gh * τ) + exp(-gh * (β - τ))) / (1 - exp(-gh * β))
    else
        return 0.0
end
end

function bare(k, p)
    if abs(k - p) > 1.0e-12
        return 4π * e0^2 * log((k + p) / abs(k - p))
    else
        return 0.0
end
end

function calcΔ(F, fdlr, kgrid, wkgrid)
    Δ0 = zeros(Float64, length(kgrid))
    Δ = zeros(Float64, (fdlr.size, length(kgrid)))
    for (ki, k) in enumerate(kgrid)
        for (qi, q) in enumerate(kgrid)
            Δ0[ki] += bare(k, q) * F[1, qi] * wkgrid[qi]  # TODO: check -F(0^-, q)==F(0^+, q)
            for (τi, τ) in enumerate(fdlr.τ)
                # Δ[τi, ki] += dW[τi, qi, ki] * F[τi, qi] * wkgrid[qi]
                Δ[τi, ki] += dH1(k, q, τ) * F[τi, qi] * wkgrid[qi]
            end
        end
    end
    return Δ0,Δ 
end

if abspath(PROGRAM_FILE) == @__FILE__
    fdlr = DLR.DLRGrid(:fermi, 10EF, β, 1e-10) 

    MaxK = 5.0 * kF
    Nk = 256
    dK = MaxK / Nk
    kgrid = LinRange(0.0, MaxK, Nk) .+ dK / 2.0
    wkgrid = [Nk / MaxK for i in 1:Nk]
    # Δ = zeros(ComplexF64, (fdlr.size, Nk)) .+ 1.0
    Δ = zeros(Float64, (fdlr.size, Nk))
    Δ0 = zeros(Float64, Nk) .+ 1.0
    F = calcF(Δ0, Δ, fdlr, kgrid)
    Δ0, Δ = calcΔ(F, fdlr, kgrid, wkgrid)
    println(Δ[1, :])
end
