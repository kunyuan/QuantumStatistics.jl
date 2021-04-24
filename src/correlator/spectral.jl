"""
Spectral representation related functions
"""
module Spectral
export kernelT, kernelΩ, density, freq2Tau, freq2MatFreq
export kernelFermiT, kernelFermiΩ, kernelBoseT, kernelBoseΩ, fermiDirac, boseEinstein
using QuadGK, Cuba
include("../fastmath.jl")
using .FastMath

"""
    kernelT(type, τ, ω, β=1.0)

Compute the imaginary-time kernel of different type.

# Arguments
- `type`: symbol :fermi, :bose, :corr
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β = 1.0`: the inverse temperature 
"""
@inline function kernelT(type::Symbol, τ::T, ω::T, β=T(1)) where {T <: AbstractFloat}
    if type == :fermi
        return kernelFermiT(τ, ω, β)
    elseif type == :bose
        return kernelBoseT(τ, ω, β)
    elseif type == :corr
        return kernelCorrT(τ, ω, β)
    else
        @error "Type $type      is not implemented!"
    end
end
"""
    kernelT(type::Symbol, τGrid::Vector{T}, ωGrid::Vector{T}, β::T=1.0) where {T<:AbstractFloat}
Compute kernel with given τ and ω grids.
"""
function kernelT(type::Symbol, τGrid::AbstractVector{T}, ωGrid::AbstractVector{T}, β=T(1)) where {T <: AbstractFloat}
    kernel = zeros(T, (length(τGrid), length(ωGrid)))
    for (τi, τ) in enumerate(τGrid)
        for (ωi, ω) in enumerate(ωGrid)
            kernel[τi, ωi] = kernelT(type, τ, ω, β)
        end
    end
    return kernel
end


"""
    kernelFermiT(τ, ω, β=1.0)

Compute the imaginary-time fermionic kernel.  Machine accuracy ~eps(g) is guaranteed``
```math
g(τ>0) = e^{-ωτ}/(1+e^{-ωβ}), g(τ≤0) = -e^{-ωτ}/(1+e^{ωβ})
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β = 1.0`: the inverse temperature 
"""
@inline function kernelFermiT(τ::T, ω::T, β=T(1)) where {T <: AbstractFloat}
    (-β < τ <= β) || error("τ=$τ must be (-β, β] where β=$β")
    if τ == T(0.0)
        τ = -eps(T)
    end
    if τ > T(0.0)
        if ω > T(0.0)
            return exp(-ω * τ) / (1 + exp(-ω * β))
        else
            return exp(ω * (β - τ)) / (1 + exp(ω * β))
        end
    else
            if ω > T(0.0)
            return -exp(-ω * (τ + β)) / (1 + exp(-ω * β))
        else
            return -exp(-ω * τ) / (1 + exp(ω * β))
        end
    end
end

"""
    kernelBoseT(τ, ω, β=1.0)

Compute the imaginary-time bosonic kernel. Machine accuracy ~eps(g) is guaranteed``
```math
g(τ>0) = e^{-ωτ}/(1-e^{-ωβ}), g(τ≤0) = -e^{-ωτ}/(1-e^{ωβ})
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β = 1.0`: the inverse temperature 
"""
@inline function kernelBoseT(τ::T, ω::T, β=T(1)) where {T <: AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    if τ == T(0.0)
        τ = -eps(T)
    end

    if τ > T(0.0)
        if ω > T(0.0)
            # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
            return exp(-ω * τ) / (-expm1(-ω * β)) 
        else
            return exp(ω * (β - τ)) / expm1(ω * β)
        end
    else
        if ω > T(0.0)
            return exp(-ω * (τ + β)) / (-expm1(-ω * β))
        else
            return exp(-ω * τ) / expm1(ω * β)
        end
    end
end

"""
    kernelCorrT(τ, ω, β=1.0)

Compute the imaginary-time kernel for correlation function ``⟨O(τ)O(0)⟩``. Machine accuracy ~eps(C) is guaranteed``
```math
K(τ) = e^{-ω|τ|}+e^{-ω(β-|τ|)}
```

# Arguments
- `τ`: the imaginary time, must be (-β, β]
- `ω`: frequency
- `β = 1.0`: the inverse temperature 
"""
@inline function kernelCorrT(τ::T, ω::T, β=T(1)) where {T <: AbstractFloat}
    (-β < τ <= β) || error("τ must be (-β, β]")
    τ=abs(τ)
    return exp(-ω*τ)+exp(-ω*(β-τ))
end


"""
    kernelΩ(type, n, ω, β=1.0)

Compute the imaginary-time kernel of different type. Assume ``k_B T/\\hbar=1``

# Arguments
- `type`: symbol :fermi, :bose, :corr
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelΩ(type::Symbol, n::Int, ω::T, β=T(1)) where {T <: AbstractFloat}
    if type == :fermi
        return kernelFermiΩ(n, ω, β)
    elseif type == :bose
        return kernelBoseΩ(n, ω, β)
    elseif type == :corr
        return kernelCorrΩ(n, ω, β)
    else
    @error "Type $type is not implemented!"
    end
end

"""
    kernelΩ(type::Symbol, nGrid::Vector{Int}, ωGrid::Vector{T}, β::T=1.0) where {T<:AbstractFloat}
Compute kernel matrix with given ωn (integer!) and ω grids.
"""
function kernelΩ(type::Symbol, nGrid::Vector{Int}, ωGrid::Vector{T}, β=T(1)) where {T <: AbstractFloat}
    kernel = zeros(Complex{T}, (length(nGrid), length(ωGrid)))
    for (ni, n) in enumerate(nGrid)
        for (ωi, ω) in enumerate(ωGrid)
        kernel[ni, ωi] = kernelΩ(type, n, ω, β)
        end
    end
    return kernel
end

"""
    kernelFermiΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the fermionic kernel with Matsubara frequency.
```math
g(iω_n) = -1/(iω_n-ω),
```
where ``ω_n=(2n+1)π/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelFermiΩ(n::Int, ω::T, β=T(1)) where {T <: AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n + 1) * π / β
    G = -1.0 / (ω_n * im - ω)
    return Complex{T}(G)
end

"""
    kernelBoseΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the bosonic kernel with Matsubara frequency.
```math
g(iω_n) = -1/(iω_n-ω),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelBoseΩ(n::Int, ω::T, β=T(1)) where {T <: AbstractFloat}
    # fermionic Matsurbara frequency
    ω_n = (2 * n) * π / β
    G = -1.0 / (ω_n * im - ω)
    if !isfinite(G)
        throw(DomainError(-1, "Got $G for the parameter $n, $ω and $β"))
    end
    return Complex{T}(G)
end

"""
    kernelCorrΩ(n::Int, ω::T, β::T) where {T <: AbstractFloat}

Compute the Matsubara-frequency kernel for a correlator ``⟨O(τ)O(0)⟩_{iω_n}``.
```math
K(iω_n) = \\frac{2ω}{ω^2+ω_n^2}(1-e^{-ωβ}),
```
where ``ω_n=2nπ/β``. The convention here is consist with the book "Quantum Many-particle Systems" by J. Negele and H. Orland, Page 95

# Arguments
- `n`: index of the Matsubara frequency
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function kernelCorrΩ(n::Int, ω::T, β=T(1)) where {T <: AbstractFloat}
    # Matsurbara-frequency correlator
    if ω<T(0.0)
        throw(DomainError("real frequency should be positive!"))
    end
    x=ω*β
    if n==0 && x<1.0e-5
        K = β*(2-x+x^2/3) #2β*(1-e^{-x})/x
    else
        ω_n = 2n * π / β
        # expm1(x)=exp(x)-1 fixes the accuracy for x-->0^+
        K = 2ω/(ω^2+ω_n^2)*(-expm1(-x))
    end
    if !isfinite(K)
        throw(DomainError(-1, "Got $K for the parameter $n, $ω and $β"))
    end
    return K
end

"""
    density(type, ω, β=1.0)

Compute the imaginary-time kernel of different type. Assume ``k_B T/\\hbar=1``

# Arguments
- `type`: symbol :fermi, :bose
- `ω`: energy 
- `β`: the inverse temperature 
"""
@inline function density(type::Symbol, ω::T, β=T(1)) where {T <: AbstractFloat}
    if type == :fermi
        return fermiDirac(ω, β)
    elseif type == :bose
        return boseEinstein(ω, β)
    else
        @error "Type $type      is not implemented!"
    end
end

"""
fermiDirac(ω)

Compute the Fermi Dirac function. Assume ``k_B T/\\hbar=1``
```math
f(ω) = 1/(e^{ωβ}+1)
```

# Arguments
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function fermiDirac(ω::T, β=T(1)) where {T <: AbstractFloat}
    x = ω * β
    if -T(50.0) < x < T(50.0)
    return 1.0 / (1.0 + exp(x))
    elseif x >= T(50.0)
        return exp(-x)
    else # x<=-50.0
        return 1.0 - exp(x)
    end
end

"""
boseEinstein(ω)

Compute the Fermi Dirac function. Assume ``k_B T/\\hbar=1``
```math
f(ω) = 1/(e^{ωβ}-1)
```

# Arguments
- `ω`: frequency
- `β`: the inverse temperature 
"""
@inline function boseEinstein(ω::T, β=T(1)) where {T <: AbstractFloat}
    # if -eps(T)<ω<eps(T)
        #     return 0.0
    # end
    n = 0.0
    x = ω * β
    if -T(50.0) < x < T(50.0)
        n = 1.0 / (exp(x) - 1.0)
    elseif x >= T(50.0)
        n = exp(-x)
    else # x<=-50.0
        n = -1.0 - exp(x)
    end
    if !isfinite(n)
throw(DomainError(-1, "Got $n for the parameter $ω and $β"))
    end
    return n
end

"""
freq2tau(type, spectral, τGrid, β=1.0, Emin=-Inf, Emax=Inf, rtol=1e-12)

Compute imaginary-time Green's function from a spectral density``
```math
G(τ>0) = ∫dω e^{-ωτ}/(1±e^{-ωβ}) S(ω)
G(τ≤0) = -∫dω e^{-ωτ}/(1±e^{ωβ}) S(ω)
```
# Arguments
- `type`: :fermi, :boson
- `integrator`: :quadgk, :vegas, :cuhre
- `spectral`: call spectral(ω) returns the spectral density
- `τGrid`: array of imaginary times to evaluate
- `β=1.0`: inverse temperature
- `Emin=-Inf`: lower bound of frequency
- `Emax=Inf`: upper bound of frequency
- `rtol=1e-12`: accuracy to achieve
"""
function freq2tau(type, integrator, spectral, τGrid, β=1.0, Emin=-Inf, Emax=Inf, rtol=1e-12)
    G = similar(τGrid)
    err = similar(τGrid)
    for (τi, τ) in enumerate(τGrid)
        f(ω) = Spectral.kernelT(type, τ / β, ω * β) * spectral(ω)
        if integrator==:quadgk
            # G[τi], err[τi] = QuadGK.quadgk(f, Emin, Emax, rtol=rtol)
            if Emin < 0.0 && Emax > 0.0
                G[τi], err[τi] = QuadGK.quadgk(f, Emin, 0.0, Emax, rtol=rtol) # integrate [Emin, 0.0] and [0.0, Emax] because the kernel is similar near 0.0
            else
                G[τi], err[τi] = QuadGK.quadgk(f, Emin, Emax, rtol=rtol)
            end
        else
            G[τi], err[τi]=FastMath.integrator1D(integrator, f, Emin, Emax, rtol)
        end
    end
    return G, err
end

"""
freq2matfreq(type, spectral, nGrid, β=1.0, Emin=-Inf, Emax=Inf, rtol=1e-12)

Compute Matubra frequency Green's function from a spectral density``
```math
G(iωn) = ∫dω -1/(iωn-ω) S(ω)
where ωn=(2n+1)π/β for fermion and ωn=2nπ/β
```
# Arguments
- `type`: :fermi, :boson
- `spectral`: call spectral(ω) returns the spectral density
- `nGrid`: array of Matsubara frequency (integer!) to evaluate
- `β=1.0`: inverse temperature
- `Emin=-Inf`: lower bound of frequency
- `Emax=Inf`: upper bound of frequency
- `rtol=1e-12`: accuracy to achieve
"""
function freq2matfreq(type, spectral, nGrid, β=1.0, Emin=-Inf, Emax=Inf, rtol=1e-12)
    G = zeros(ComplexF64, length(nGrid))
    err = similar(G)
        for (ni, n) in enumerate(nGrid)
        f(ω) = Spectral.kernelΩ(type, n, ω * β) * spectral(ω)
        if Emin < 0.0 && Emax > 0.0
            G[ni], err[ni] = QuadGK.quadgk(f, Emin, 0.0, Emax, rtol=rtol) # integrate [Emin, 0.0] and [0.0, Emax] because the kernel is similar near 0.0
        else
            G[ni], err[ni] = QuadGK.quadgk(f, Emin, Emax, rtol=rtol)
        end
    end
    return G, err
end 
end
