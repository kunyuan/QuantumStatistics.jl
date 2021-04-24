"""
discrete Lehmann representation for imaginary-time/Matsubara-freqeuncy correlator
"""
module DLR
export DLRGrid, dlr
export tau2dlr, tau2matfreq, matfreq2dlr, matfreq2tau, tau2matfreq, matfreq2tau
using DelimitedFiles, LinearAlgebra
# include("spectral.jl")
using ..Spectral
include("builder.jl")


"""
struct DLRGrid

    DLR grids for imaginary-time/Matsubara frequency correlators

#Members:
- `type`: symbol :fermi, :bose, :corr
- `Euv` : the UV energy scale of the spectral density 
- `β` : inverse temeprature
- `Λ`: cutoff = UV Energy scale of the spectral density * inverse temperature
- `rtol`: tolerance absolute error
- `size` : number of DLR basis
- `ω` : selected representative real-frequency grid
- `n` : selected representative Matsubara-frequency grid (integer)
- `ωn` : (2n+1)π/β
- `τ` : selected representative imaginary-time grid
"""
struct DLRGrid
    type::Symbol
    Euv::Float64
    β::Float64
    Λ::Float64
    rtol::Float64

    # dlr grids
    size::Int # rank of the dlr representation
    ω::Vector{Float64}
    n::Vector{Int} # integers, (2n+1)π/β gives the Matsubara frequency
    ωn::Vector{Float64} # (2n+1)π/β
    τ::Vector{Float64}

    """
    function DLRGrid(type, Euv, β, rtol=1e-12)

        Create DLR grids
    """
    function DLRGrid(type, Euv, β, rtol=1e-12)
        Λ = Euv * β # dlr only depends on this dimensionless scale
        @assert rtol > 0.0 "eps=$eps is not positive and nonzero!"
        @assert 0 < Λ <= 1000000 "Energy scale $Λ must be in (0, 1000000)!"
        if Λ < 100 
            Λ = Int(100)
        else
            Λ = 10^(Int(ceil(log10(Λ)))) # get smallest n so that Λ<10^n
        end
        rtolpower = Int(floor(log10(rtol))) # get the biggest n so that rtol>1e-n
        if abs(rtolpower) < 4
            rtolpower = 4
        end
        
        filename = string(@__DIR__, "/basis/$(string(type))/dlr$(Λ)_1e$(rtolpower).dlr")

        if isfile(filename) == false
            error("$filename doesn't exist. DLR basis hasn't been generated.")
        end

        grid = readdlm(filename)

        ω = grid[:, 2] / β
        n = Int.(grid[:, 4])
        if type==:fermi
            ωn = @. (2n + 1.0) * π / β
        elseif type ==:corr || type==:boson
            ωn = @. 2n * π / β
        else
            error("$type not implemented!")
        end
        τ = grid[:, 3] * β
        return new(type, Euv, β, Λ, rtol, length(ω), ω, n, ωn, τ)
    end
end

function _tensor2matrix(tensor, axis)
    # internal function to move the axis dim to the first index, then reshape the tensor into a matrix
    dim = length(size(tensor))
    n1 = size(tensor)[axis]
    partialsize = deleteat!(collect(size(tensor)), axis) # the size of the tensor except the axis-th dimension
    n2 = reduce(*, partialsize)
    # println("working on size ", size(tensor))
    # println(axis)
    permu = [i for i in 1:dim]
    permu[1], permu[axis] = axis, 1
    ntensor = permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created even for axis=1
    ntensor = reshape(ntensor, (n1, n2)) # no copy is created
    return ntensor, partialsize
end

function _matrix2tensor(mat, partialsize, axis)
    # internal function to reshape matrix to a tensor, then swap the first index with the axis-th dimension
    @assert size(mat)[2] == reduce(*, partialsize) # total number of elements of mat and the tensor must match
    tsize = vcat(size(mat)[1], partialsize)
    tensor = reshape(mat, Tuple(tsize))
    dim = length(partialsize) + 1
    permu = [i for i in 1:dim]
    permu[1], permu[axis] = axis, 1
    return permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created even for axis=1
end

"""
function tau2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)

    imaginary-time domain to DLR representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `green` : green's function in imaginary-time domain
- `axis`: the imaginary-time axis in the data `green`
- `rtol`: tolerance absolute error
"""
function tau2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    τGrid = dlrGrid.τ
    ωGrid = dlrGrid.ω

    kernel = kernelT(type, τGrid, ωGrid, dlrGrid.β)
    # kernel, ipiv, info = LAPACK.getrf!(Float64.(kernel)) # LU factorization
    kernel, ipiv, info = LAPACK.getrf!(kernel) # LU factorization

    g, partialsize = _tensor2matrix(green, axis)

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g #solve green=kernel*coeff
    # println("coeff: ", maximum(abs.(coeff)))

    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2tau(type, dlrcoeff, dlrGrid::DLRGrid, τGrid; axis=1)

    DLR representation to imaginary-time representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `dlrcoeff` : DLR coefficients
- `dlrGrid` : DLRGrid
- `τGrid` : expected fine imaginary-time grids ∈ (0, β]
- `axis`: imaginary-time axis in the data `dlrcoeff`
- `rtol`: tolerance absolute error
"""
function dlr2tau(type, dlrcoeff, dlrGrid::DLRGrid, τGrid; axis=1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    @assert all(τGrid .> 0.0) && all(τGrid .<= dlrGrid.β)
    β=dlrGrid.β
    ωGrid=dlrGrid.ω

    kernel = kernelT(type, τGrid, ωGrid, dlrGrid.β)

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function matfreq2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)

    Matsubara-frequency representation to DLR representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `green` : green's function in Matsubara-frequency domain
- `axis`: the Matsubara-frequency axis in the data `green`
- `rtol`: tolerance absolute error
"""
function matfreq2dlr(type, green, dlrGrid::DLRGrid; axis=1, rtol=1e-12)
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    nGrid = dlrGrid.n
    ωGrid = dlrGrid.ω

    kernel = kernelΩ(type, nGrid, ωGrid, dlrGrid.β)
    # kernel, ipiv, info = LAPACK.getrf!(Complex{Float64}.(kernel)) # LU factorization
    kernel, ipiv, info = LAPACK.getrf!(kernel) # LU factorization

    g, partialsize = _tensor2matrix(green, axis)

    coeff = LAPACK.getrs!('N', kernel, ipiv, g) # LU linear solvor for green=kernel*coeff
    # coeff = kernel \ g # solve green=kernel*coeff
    # coeff/=dlrGrid.Euv

    return _matrix2tensor(coeff, partialsize, axis)
end

"""
function dlr2matfreq(type, dlrcoeff, dlrGrid::DLRGrid, nGrid, β=1.0; axis=1)

    DLR representation to Matsubara-frequency representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `dlrcoeff` : DLR coefficients
- `dlrGrid` : DLRGrid
- `nGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `axis`: Matsubara-frequency axis in the data `dlrcoeff`
- `rtol`: tolerance absolute error
"""
function dlr2matfreq(type, dlrcoeff, dlrGrid::DLRGrid, nGrid, β=1.0; axis=1)
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    ωGrid = dlrGrid.ω

    kernel = kernelΩ(type, nGrid, ωGrid, dlrGrid.β) 

    coeff, partialsize = _tensor2matrix(dlrcoeff, axis)

    G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]

    return _matrix2tensor(G, partialsize, axis)
end

"""
function tau2matfreq(type, green, dlrGrid, nGrid; axis=1, rtol=1e-12)

    Fourier transform from imaginary-time to Matsubara-frequency using the DLR representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `green` : green's function in imaginary-time domain
- `dlrGrid` : DLRGrid
- `nGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `axis`: the imaginary-time axis in the data `green`
- `rtol`: tolerance absolute error
"""
function tau2matfreq(type, green, dlrGrid, nGrid; axis=1, rtol=1e-12)
    coeff = tau2dlr(type, green, dlrGrid; axis=axis, rtol=rtol)
    return dlr2matfreq(type, coeff, dlrGrid, nGrid, axis=axis)
end

"""
function matfreq2tau(type, green, dlrGrid, τGrid; axis=1, rtol=1e-12)

    Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `type`: symbol :fermi, :bose, :corr
- `green` : green's function in Matsubara-freqeuncy repsentation
- `dlrGrid` : DLRGrid
- `τGrid` : expected fine imaginary-time grids
- `axis`: Matsubara-frequency axis in the data `green`
- `rtol`: tolerance absolute error
"""
function matfreq2tau(type, green, dlrGrid, τGrid; axis=1, rtol=1e-12)
    coeff = matfreq2dlr(type, green, dlrGrid; axis=axis, rtol=rtol)
    return dlr2tau(type, coeff, dlrGrid, τGrid, axis=axis)
end

end