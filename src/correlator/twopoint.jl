"""
Provide N-body response and correlation functions
"""
module TwoPoint

export freePropagatorT, freePropagatorΩ
export freePolarizationT
using ..Spectral
using Cuba

"""
    freePropagatorT(type, τ, ω, β)

Imaginary-time propagator.

# Arguments
- `type`: symbol :fermi, :bose
- `τ`: the imaginary time, must be (-β, β]
- `ω`: dispersion ϵ_k-μ
- `β = 1.0`: the inverse temperature 
"""
@inline function freePropagatorT(type, τ, ω, β)
    return kernelT(type, τ, ω, β)
end

"""
    freePropagatorΩ(type, n, ω, β=1.0)

Matsubara-frequency kernel of different type

# Arguments
- `type`: symbol :fermi, :bose, :corr
- `n`: index of the Matsubara frequency
- `ω`: dispersion ϵ_k-μ
- `β`: the inverse temperature 
"""
@inline function freePropagatorΩ(type, n::Int, ω, β)
    return kernelΩ(type, n, ω, β)
end

@inline function freeFermiDoS(dim, kF, m, spin)
    if dim == 3
        return spin * m * kF / 2 / π^2
    else
        error("Dimension $dim not implemented!")
        # return spin/4/π
    end
end

# function LindhardΩn(dim, q, ω, β, kF, m, spin)
#     q < 0.0 && (q = -q) # Lindhard function is q symmetric

#     q2 = q^2
#     kFq = 2kF * q
#     ωn = 2π * n / β
#     D = 1 / (8kF * q)
#     NF = freeFermiDoS(dim, kF, m, spin)

#     # if ωn<=20*(q2+kFq)/(2m)
#         # careful for small q or large w
#     iw = ωn * im
#     wmq2 = iw * 2m - q^2
#     wpq2 = iw * 2m + q^2
#     C1 = log(wmq2 - kFq) - log(wmq2 + kFq)
#     C2 = log(wpq2 - kFq) - log(wpq2 + kFq)
#     res = real(-NF / 2 * (1 - D * (wmq2^2 / q^2 - 4 * kF^2) * C1 + D * (wpq2^2 / q^2 - 4 * kF^2) * C2))
#     # else
#     #     b2 = q2 * ( q2 + 12/5 * kF^2 )
#     #     c = 2*EF*kF*q2/(3*pi**2)
#     #     res = -c/(w**2 + b2)
#     # end
#     return res
# end

"""
    LindhardΩnFiniteTemperature(dim::Int, q::T, n::Int, μ::T, kF::T, β::T, m::T, spin) where {T <: AbstractFloat}

Compute the polarization function of free electrons at a given frequency. Relative Accuracy is about ~ 1e-6

# Arguments
- `dim`: dimension
- `q`: external momentum, q<1e-4 will be treated as q=0 
- `n`: externel Matsubara frequency, ωn=2π*n/β
- `μ`: chemical potential
- `kF`: Fermi momentum 
- `β`: inverse temperature
- `m`: mass
- `spin` : number of spins
"""
@inline function LindhardΩnFiniteTemperature(dim::Int, q::T, n::Int, μ::T, kF::T, β::T, m::T, spin) where {T <: AbstractFloat}
    if q < 0.0
        q = -q
    end

    if q / kF < 1.0e-10
        q = 1.0e-10 * kF
    end

    function polar(k)
        phase = T(1.0)
        if dim == 3
            phase *= k^2 / (4π^2)
        else
            error("not implemented")
        end
        ω = 2π * n / β
        ϵ = β * (k^2 - μ) / (2m)

        p = phase * fermiDirac(ϵ) * m / k / q * log(((q^2 - 2k * q)^2 + 4m^2 * ω^2) / ((q^2 + 2k * q)^2 + 4m^2 * ω^2)) * spin

        if isnan(p)
            println("warning: integrand at ω=$ω, q=$q, k=$k is NaN!")
        end
        # println(p)
        return p
    end

    function integrand(x, f)
        # x[1]:k
        f[1] = polar(x[1] / (1 - x[1])) / (1 - x[1])^2
    end

    result, err = Cuba.cuhre(integrand, 2, 1, rtol=1.0e-10)
    # result, err = Cuba.vegas(integrand, 1, 1, rtol=rtol)
    return result[1], err[1]
end
end