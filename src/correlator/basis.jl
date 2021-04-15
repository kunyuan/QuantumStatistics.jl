"""
Basis functions for the correlator
"""
module Basis
export tauGrid
using DelimitedFiles, LinearAlgebra
# include("spectral.jl")
using ..Spectral

function dlrGrid(type, Euv, β=1.0, eps=1e-10)
    Λ = Euv * β # dlr only depends on this dimensionless scale
    @assert eps > 0.0 "eps=$eps is not positive and nonzero!"
    @assert 0 < Λ <= 1000000 "Energy scale $Λ must be in (0, 1000000)!"
    if Λ < 100 
        Λ = Int(100)
    else
        Λ = 10^(Int(ceil(log10(Λ)))) # get smallest n so that Λ<10^n
    end
    epspower = Int(floor(log10(eps))) # get the biggest n so that eps>1e-n
    if abs(epspower) < 4
        epspower = 4
    end

    if type == :fermi
        filename = string(@__DIR__, "/dlr/basis/fermi/dlr$(Λ)_1e$(epspower).dat")
        # filename = string(@__DIR__, "/basis/dlr_fermi/dlr$(Λ)_1e$(epspower).dat")
        println(filename)
        grid = readdlm(filename)

        ω = grid[:, 2] / β
        ωn = Int.(grid[:, 4])
        tgrid = [((t >= 0.0) ? t : 1.0 + t) for t in grid[:, 3]]
        τ = sort(tgrid * β)
        return Dict(:ω => ω, :τ => τ, :ωn => ωn)
    else
        @error "Not implemented!"
    end
end

function tau2dlr(type, green, dlrGrid, β=1.0; axis=1, rtol=1e-12)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    τGrid = dlrGrid[:τ]
    ωGrid = dlrGrid[:ω]
    kernel = kernelT(type, τGrid, ωGrid, β)
    # kernel, ipiv, info = LAPACK.getrf!(Float64.(kernel)) # LU factorization
    kernel, ipiv, info = LAPACK.getrf!(kernel) # LU factorization

    if axis == 1
        g = copy(green)
    else
        g = permutedims(green, [axis, 1])
    end

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g #solve green=kernel*coeff
    # println("coeff: ", maximum(abs.(coeff)))

    if axis == 1
        return coeff
    else
        ############ test ######################
        # kernel = kernelT(type, τGrid, ωGrid, β)
        # gfitted = kernel * coeff
        # println("LU error: ", maximum(abs.(gfitted' - green)))
        ##########################################
        return permutedims(coeff, [axis, 1])
    end
end

function dlr2tau(type, dlrcoeff, dlrGrid, τGrid, β=1.0; axis=1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    kernel = kernelT(type, τGrid, dlrGrid[:ω], β)
    if axis == 1
        coeff = dlrcoeff
    else
        coeff = permutedims(dlrcoeff, [axis, 1])
    end

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    if axis == 1
        return G
    else
        return permutedims(G, [axis, 1])
    end
end

function matfreq2dlr(type, green, dlrGrid, β=1.0; axis=1, rtol=1e-12)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    nGrid = dlrGrid[:ωn]
    ωGrid = dlrGrid[:ω]
    # kernel = kernelΩ(type, nGrid, ωGrid, β) / β 
    kernel = kernelΩ(type, nGrid, ωGrid, β)
    kernel, ipiv, info = LAPACK.getrf!(Complex{Float64}.(kernel)) # LU factorization

    if axis == 1
        g = copy(green)
    else
        g = permutedims(green, [axis, 1])
    end

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g # solve green=kernel*coeff

    if axis == 1
        return coeff
    else
        return permutedims(coeff, [axis, 1])
    end
end

function dlr2matfreq(type, dlrcoeff, dlrGrid, nGrid, β=1.0; axis=1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    # kernel = kernelΩ(type, nGrid, dlrGrid[:ω], β) / β 
    kernel = kernelΩ(type, nGrid, dlrGrid[:ω], β) 
    if axis == 1
        coeff = dlrcoeff
    else
        coeff = permutedims(dlrcoeff, [axis, 1])
    end

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    if axis == 1
        return G
    else
        return permutedims(G, [axis, 1])
    end
end

function tau2matfreq(type, green, dlrGrid, nGrid, β=1.0; axis=1, rtol=1e-12)
    coeff = tau2dlr(type, green, dlrGrid, β; axis=axis, rtol=rtol)
    return dlr2matfreq(type, coeff, dlrGrid, nGrid, β, axis=axis)
end

function matfreq2tau(type, green, dlrGrid, τGrid, β=1.0; axis=1, rtol=1e-12)
    coeff = matfreq2dlr(type, green, dlrGrid, β; axis=axis, rtol=rtol)
    return dlr2tau(type, coeff, dlrGrid, τGrid, β, axis=axis)
end

end