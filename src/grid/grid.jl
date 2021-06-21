module Grid

# export Log, UniformGrid, tauGrid, fermiKGrid, boseKGrid

using StaticArrays

@enum GridType LOG UNIFORM UNILOG

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

struct UniLog{T <: AbstractFloat}
    bound::SVector{2,T}
    idx::SVector{2,T}
    isopen::SVector{2,Bool}
    M::Int
    N::Int
    λ::T
    d2s::Bool

    function UniLog{T}(bound, init, minterval::T,M::Int,N::Int, dense2sparse::Bool,isopen = @SVector[false,true]) where {T}
        @assert N*minterval<bound[2]-bound[1]
        Nidx = (M+1)*N
        if isopen[2]==true
            Nidx = Nidx-1
        end
        idx = @SVector[init, init + Nidx ]
        λ = (minterval*N/(bound[2]-bound[1]))^(1.0/M)
        return new{T}(bound, idx,isopen, M,N, λ, dense2sparse)
    end
end

function _grid(l::UniLog{T}, i) where {T}
    head = !l.isopen[1] ? l.idx[1] : (l.idx[1]-1)
    tail = !l.isopen[2] ? l.idx[2] : (l.idx[2]+1)
    if l.d2s
        i_n = (i - head)%l.N;
        i_m = (i - head)÷l.N;
        if i_m==0
            return l.bound[1]+l.λ^l.M/l.N*(l.bound[2]-l.bound[1])*i_n
        else
            return l.bound[1]+l.λ^(l.M+1-i_m)*(l.bound[2]-l.bound[1])+(l.λ^(l.M-i_m)-l.λ^(l.M-i_m+1))/l.N*i_n*(l.bound[2]-l.bound[1])
        end
    end
    if !l.d2s
        i_n = (tail - i)%l.N;
        i_m = (tail - i)÷l.N;
        if i_m==0
            return l.bound[2]-l.λ^(l.M)/l.N*(l.bound[2]-l.bound[1])*i_n
        else
            return l.bound[2]-l.λ^(l.M+1-i_m)*(l.bound[2]-l.bound[1])-(l.λ^(l.M-i_m)-l.λ^(l.M-i_m+1))/l.N*i_n*(l.bound[2]-l.bound[1])
        end
    end
end

function _floor(l::UniLog{T}, x::T) where {T}
    head = !l.isopen[1] ? l.idx[1] : (l.idx[1]-1)
    tail = !l.isopen[2] ? l.idx[2] : (l.idx[2]+1)
    length = l.bound[2] - l.bound[1]

    norm = l.d2s ? ((x - l.bound[1])/length) : ((l.bound[2] - x)/length)
    if norm <= 0
        pos = l.d2s ? l.idx[1] : l.idx[2]
        return Base.floor(Int,pos)
    elseif norm >=1
        pos = l.d2s ? l.idx[2] : l.idx[1]
        return Base.floor(Int,pos)
    end

    i_m= Base.floor(log(norm)/log(l.λ))
    if i_m>=l.M
        if l.d2s
            result = Base.floor(head+norm/l.λ^(l.M)*l.N)
            return (l.isopen[1]&&result==head) ? (result+1) : result
        else
            result = Base.floor(tail-norm/l.λ^(l.M)*l.N)
            return (l.isopen[2]&&result==tail) ? (result-1) : result
        end
    end
    i_n= Base.floor((norm-l.λ^(i_m+1))/(l.λ^(i_m)-l.λ^(i_m+1))*l.N)
    if l.d2s
        result = Base.floor(head+(l.M-i_m)*l.N+i_n)
        return (l.isopen[2]&&result==tail) ? result-1 : result
    else
        result = Base.floor((tail-(l.M-i_m)*l.N-i_n)-1)
        return (l.isopen[1]&&result==head) ? result+1 : result
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


struct UniLogs{T<:AbstractFloat,SIZE,SEG}
    grid::MVector{SIZE,T}
    size::Int
    head::T
    tail::T
    unilogs::SVector{SEG,UniLog{T}}
    segment::SVector{SEG,T} # ends of each segments
    segindex::SVector{SEG,Int}
    isopen::SVector{2,Bool}

    function UniLogs{T,SIZE,SEG}(bounds, minterval::T,M::Int,N::Int, Isopen = @SVector[true,true], issparse = @SVector[false,false]) where {T<:AbstractFloat,SIZE,SEG}
        @assert SEG > 0 
        size = (M+1)*N*SEG + 1
        @assert SIZE == size 

        grid, segment,segindex = [], [],[]
        unilogs = []
        if issparse[1]==false
            for s = 1:SEG
                if s%2==1
                    bound = @SVector[ bounds[(s+1)÷2], (bounds[(s+1)÷2+1]+bounds[(s+1)÷2])/2]
                    if s == SEG && issparse[2]
                        bound = @SVector[ bounds[(s+1)÷2],bounds[(s+1)÷2+1]]
                    end
                    init = 1 + (M+1)*N*(s-1)
                    push!(segment,bound[2])
                    push!(segindex, 1 + (M+1)*N*(s))
                    isopen = @SVector[false, true]
                    if s == SEG && issparse[2]
                        isopen = @SVector[false, false]
                    end
                    g = UniLog{T}(bound,init,minterval,M,N,true,isopen)
                    push!(unilogs,g)
                    for idx=g.idx[1]:g.idx[2]
                        push!(grid, _grid(g, idx))
                    end
                else
                    bound = @SVector[ (bounds[s÷2+1]+bounds[s÷2])/2,bounds[s÷2+1]]
                    init = 1 + (M+1)*N*(s-1)
                    push!(segment, bound[2])
                    push!(segindex, 1 + (M+1)*N*(s))
                    isopen = @SVector[false,s==SEG ? false : true]
                    g = UniLog{T}(bound,init,minterval,M,N,false,isopen)
                    push!(unilogs,g)
                    for idx=g.idx[1]:g.idx[2]
                        push!(grid, _grid(g, idx))
                    end
                end
            end
        else
            for s = 1:SEG
                if s%2==1
                    bound = @SVector[(bounds[(s+1)÷2+1]+bounds[(s+1)÷2])/2,bounds[(s+1)÷2+1]]
                    if s == 1
                        bound = @SVector[ bounds[1], bounds[2]]
                    end
                    init = 1 + (M+1)*N*(s-1)
                    push!(segment,bound[2])
                    push!(segindex, 1 + (M+1)*N*(s))
                    isopen = @SVector[false, s==SEG ? false : true]
                    g = UniLog{T}(bound,init,minterval,M,N,false,isopen)
                    push!(unilogs,g)
                    for idx=g.idx[1]:g.idx[2]
                        push!(grid, _grid(g, idx))
                    end
                else
                    bound = @SVector[bounds[s÷2+1], (bounds[s÷2+1]+bounds[s÷2+2])/2]
                    if s == SEG && issparse[2]
                        bound = @SVector[bounds[s÷2+1], bounds[(s+2)÷2+1]]
                    end
                    init = 1 + (M+1)*N*(s-1)
                    push!(segment, bound[2])
                    push!(segindex, 1 + (M+1)*N*(s))
                    isopen = @SVector[false,s==SEG ? false : true]
                    g = UniLog{T}(bound,init,minterval,M,N,true,isopen)
                    push!(unilogs,g)
                    for idx=g.idx[1]:g.idx[2]
                        push!(grid, _grid(g, idx))
                    end
                end
            end
        end

        head, tail = grid[1], grid[end]
        Isopen[1] && (grid[1] += eps(T) * 1e4)
        Isopen[2] && (grid[end] -= eps(T) * 1e4)
        checkOrder(grid)
        return new{T,size,SEG}(grid, size, head, tail,unilogs, segment,segindex, Isopen)
    end
end

function Base.floor(grid::UniLogs{T,SIZE,SEG}, x) where {T,SIZE,SEG}
#    (grid.head <= x <= grid.tail) || error("$x is out of the uniform grid range!")
    segment = grid.segment
    seg = 0
    for i=1:SEG
        if x<segment[i] && seg==0
            seg = i
        end
    end
    if seg == 0
        seg = SEG
    end
    result = _floor(grid.unilogs[seg],x)
    return Base.floor(Int,result==grid.size ? result-1 : result)
end

Base.getindex(grid::UniLogs, i) = grid.grid[i]
Base.firstindex(grid::UniLogs) = 1
Base.lastindex(grid::UniLogs) = grid.size


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
    tauUL(β, minterval, M::Int, N::Int, type = Float64)

Create a unilog Grid for the imaginary time, which is densest near the 0 and β
the grid has 2 segments, 2*(M+1)*N+1 points.

#Arguments:
- `β`: inverse temperature
- minterval: the minimal interval of the grid
- M: (M+1) is the number of the 'log' grid of each segment of the grids, M>=0
- N: each interval of the 'log' grid is further divided uniformly into N part, N>=1
"""
@inline function tauUL(β, minterval, M::Int,N::Int, type=Float64)
    seg = 2
    size = (M+1)*N*seg+1
    bounds = @SVector[0.0,β]
    return UniLogs{Float64,size,seg}(bounds,minterval,M,N)
end

@inline function tauUL(para, M::Int,N::Int, type=Float64)
    return tauUL(para.β, para.τ_min, M, N, type)
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
    fermiKUL(Kf, maxK, minterval, M, N,  type = Float64)

Create a unilog fermionic K Grid, which is densest near the Fermi momentum `k_F`
the grid has 2 segments, 2*(M+1)*N+1 points.

#Arguments:
- Kf: Fermi momentum
- maxK: the upper bound of the grid
- minterval: the minimal interval of the grid
- M: (M+1) is the number of the 'log' grid of each segment of the grids, M>=0
- N: each interval of the 'log' grid is further divided uniformly into N part, N>=1
"""

@inline function fermiKUL(Kf, maxK, minterval, M::Int,N::Int, type=Float64)
    seg = 2
    size = (M+1)*N*seg+1
    bounds = @SVector[0.0,Kf,maxK]
    return UniLogs{Float64,size,seg}(bounds,minterval,M,N,[true,false],[true,true])
end

@inline function fermiKUL(para, M::Int,N::Int, type=Float64)
    return fermiKUL(para.kF, para.k_max, para.k_min, M, N, type)
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

"""
    boseKUL(Kf, maxK, minterval, M, N,  type = Float64)

Create a unilog bosonic K Grid, which is densest near the momentum `0` and `2k_F`
the grid has 3 segments, 3*(M+1)*N+1 points.

#Arguments:
- Kf: Fermi momentum
- maxK: the upper bound of the grid
- minterval: the minimal interval of the grid
- M: (M+1) is the number of the 'log' grid of each segment of the grids, M>=0
- N: each interval of the 'log' grid is further divided uniformly into N part, N>=1
"""
@inline function boseKUL(
    Kf,
    maxK,
    minterval,
    M,
    N,
    type=Float64
)
    seg = 3
    size = (M+1)*N*seg+1
    bounds = @SVector[0.0,2Kf,maxK]
    return UniLogs{Float64,size,seg}(bounds,minterval,M,N,[true,false],[false,true])
end

@inline function boseKUL(para, M::Int,N::Int, type=Float64)
    return boseKUL(para.kF, para.k_max, para.k_min, M, N, type)
end


include("interpolate.jl")

end
