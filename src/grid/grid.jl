module Grid

# export Log, UniformGrid, tauGrid, fermiKGrid, boseKGrid

using StaticArrays: SVector, MVector

@enum GridType LOG UNIFORM

struct Coeff{T <: AbstractFloat}
    bound::SVector{2,T}
    idx::SVector{2,T}
    λ::T
    a::T
    b::T

    function Coeff{T}(bound, idx, λ, dense2sparse::Bool) where {T}
        λ /= abs(idx[2] - idx[1])
        if dense2sparse == false
            bound = (bound[2], bound[1])
            idx = (idx[2], idx[1])
            λ = -λ
        end
        _l1, _l2 = T(1.0), exp(λ * (idx[2] - idx[1]))
        b = (bound[2] - bound[1]) / (_l2 - _l1)
        a = (bound[1] * _l2 - bound[2] * _l1) / (_l2 - _l1)
        return new{T}(bound, idx, λ, a, b)
    end
end

function _grid(l::Coeff{T}, pos) where {T}
    return l.a + l.b * exp(l.λ * (T(pos) - l.idx[1]))
end

function _floor(l::Coeff{T}, x::T) where {T}
    pos = l.idx[1] + log((x - l.a) / l.b) / l.λ
    return Base.floor(Int, pos)
end

function checkOrder(grid)
    for idx = 2:length(grid)
        @assert grid[idx - 1] < grid[idx] "The grid at $idx is not in the increase order: \n$grid"
    end
end

struct Log{T,SIZE,SEG} # create a log grid of the type T with SIZE grids and SEG of segments
    grid::MVector{SIZE,T}
    size::Int
    head::T
    tail::T
    segment::SVector{SEG,Int} # ends of each segments
    coeff::SVector{SEG,Coeff{T}}
    isopen::SVector{2,Bool}

    function Log{T,SIZE,SEG}(coeff, range, isopen) where {T <: AbstractFloat,SIZE,SEG}
        @assert SIZE > 1 "Size must be large than 1"
        for ri = 2:SEG
            @assert range[ri - 1][end] + 1 == range[ri][1] "ranges must be connected to each other"
        end
        @assert range[1][1] == 1 "ranges should start with the idx 1"
        @assert range[end][end] == SIZE "ranges ends with $(range[end][end]), expected $SIZE"

        grid, segment = [], []
        for s = 1:SEG
            push!(segment, range[s][end])
            for idx in range[s]
                push!(grid, _grid(coeff[s], idx))
            end
        end
        head, tail = grid[1], grid[end]
        isopen[1] && (grid[1] += eps(T) * 1e4)
        isopen[2] && (grid[end] -= eps(T) * 1e4)
        checkOrder(grid)
        return new{T,SIZE,SEG}(grid, SIZE, head, tail, segment, coeff, isopen)
    end
end

function Base.floor(grid::Log{T,SIZE,2}, x) where {T,SIZE}
    (grid.head <= x <= grid.tail) || error("$x is out of the uniform grid range!")
    segment = grid.segment
    if x < grid[2]
        return 1
    elseif x < grid[segment[1]]
        return _floor(grid.coeff[1], x)
    elseif x < grid[segment[1] + 1]
        return segment[1]
    elseif x < grid[end - 1]
        return _floor(grid.coeff[2], x)
    else
        return SIZE - 1
    end
end

function Base.floor(grid::Log{T,SIZE,3}, x) where {T,SIZE}
    (grid.head <= x <= grid.tail) || error("$x is out of the uniform grid range!")
    segment = grid.segment
    if x < grid[2]
        return 1
    elseif x < grid[segment[1]]
        return _floor(grid.coeff[1], x)
    elseif x < grid[segment[1] + 1]
        return segment[1]
    elseif x < grid[segment[2]]
        return _floor(grid.coeff[2], x)
    elseif x < grid[segment[2] + 1]
        return segment[2]
    elseif x < grid[end - 1]
        return _floor(grid.coeff[3], x)
    else
        return SIZE - 1
    end
end

Base.getindex(grid::Log, i) = grid.grid[i]
Base.firstindex(grid::Log) = 1
Base.lastindex(grid::Log) = grid.size

"""
    Uniform{Type,SIZE}

Create a uniform Grid with a given type and size

# Member:
- `β`: inverse temperature
- halfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)
- size: the Grid size
- grid: vector stores the grid
- size: size of the grid vector
- head: grid head
- tail: grid tail
- δ: distance between two grid elements
- isopen: if isopen[1]==true, then grid[1] will be slightly larger than the grid head. Same for the tail.
"""
struct Uniform{T,SIZE}
    grid::SVector{SIZE,T}
    size::Int
    head::T
    tail::T
    δ::T
    isopen::SVector{2,Bool}

    """
        Uniform{Type,SIZE}(head, tail, isopen)

    Create a uniform Grid with a given type and size

    # Arguments:
     - head: the starting point of the grid
     - tail: the end of the grid
     - isopen: if isopen[1]==true, then grid[1]=head+eps; If isopen[2]==true, then grid[2]=tail-eps. Otherwise, grid[1]==head / grid[2]==tail
    """
    function Uniform{T,SIZE}(head, tail, isopen) where {T <: AbstractFloat,SIZE}
        @assert SIZE > 1 "Size must be large than 1"
        grid = Array(LinRange(T(head), T(tail), SIZE))
        isopen[1] && (grid[1] += eps(T))
        isopen[2] && (grid[end] -= eps(T))
        return new{T,SIZE}(grid, SIZE, head, tail, (tail - head) / (SIZE - 1), isopen)
end
end

function Base.floor(grid::Uniform, x)

    (grid.head <= x <= grid.tail) || error("$x is out of the uniform grid range!")

    if grid[2] <= x < grid[end - 1]
        return floor(Int, (x - grid.head) / grid.δ + 1)
    elseif x < grid[2]
        return 1
    else
        return grid.size - 1
end
end

Base.getindex(grid::Uniform, i) = grid.grid[i]
Base.firstindex(grid::Uniform) = 1
Base.lastindex(grid::Uniform) = grid.size


"""
    tau(β, halfLife, size::Int, type = Float64)

Create a logarithmic Grid for the imaginary time, which is densest near the 0 and β

#Arguments:
- `β`: inverse temperature
- halfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)
- size: the Grid size
"""
@inline function tau(β, halfLife, size::Int, type=Float64)
    size = Int(size)
    c1 = Grid.Coeff{type}([0.0, 0.5β], [1.0, 0.5size + 0.5], 1.0 / halfLife, true)
    r1 = 1:Int(0.5size)

    c2 = Grid.Coeff{type}([0.5β, β], [0.5size + 0.5, size], 1.0 / halfLife, false)
    r2 = (Int(0.5size) + 1):size
    return Log{type,size,2}([c1, c2], [r1, r2], [true, true])
end

"""
    fermiK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), type = Float64)

Create a logarithmic fermionic K Grid, which is densest near the Fermi momentum ``k_F``

#Arguments:
- Kf: Fermi momentum
- maxK: the upper bound of the grid
- halfLife: the grid is densest in the range (Kf-halfLife, Kf+halfLife)
- size: the Grid size
- kFi: index of Kf
"""
@inline function fermiK(Kf, maxK, halfLife, size::Int, kFi=floor(Int, 0.5size), type=Float64)
    size = Int(size)
    c1 = Grid.Coeff{type}([0.0, Kf], [1.0, kFi], 1.0 / halfLife, false)
    r1 = 1:kFi

    c2 = Grid.Coeff{type}([Kf, maxK], [kFi - 1.0, size], 1.0 / halfLife, true)
    r2 = kFi + 1:size
    return Log{type,size,2}([c1, c2], [r1, r2], [true, false])
end

"""
    boseK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), twokFi = floor(Int, 2size / 3), type = Float64)

Create a logarithmic bosonic K Grid, which is densest near the momentum `0` and `2k_F`

#Arguments:
- Kf: Fermi momentum
- maxK: the upper bound of the grid
- halfLife: the grid is densest in the range (0, Kf+halfLife) and (2Kf-halfLife, 2Kf+halfLife)
- size: the Grid size
- kFi: index of Kf
- twokFi: index of 2Kf
"""
@inline function boseK(
    Kf,
    maxK,
    halfLife,
    size::Int,
    kFi=floor(Int, size / 3),
    twokFi=floor(Int, 2size / 3),
    type=Float64,
)
    size = Int(size)
    λ = 1.0 / halfLife
    c1 = Grid.Coeff{type}([0.0, Kf], [1.0, kFi], λ, true)
    r1 = 1:kFi

    c2 = Grid.Coeff{type}([Kf, 2.0 * Kf], [kFi - 1.0, twokFi], λ, false)
    r2 = kFi + 1:twokFi

    c3 = Grid.Coeff{type}([2.0 * Kf, maxK], [twokFi - 1.0, size], λ, true)
    r3 = twokFi + 1:size

    K = Grid.Log{type,size,3}([c1, c2, c3], [r1, r2, r3], [true, false])
return K
end

include("interpolate.jl")

end
