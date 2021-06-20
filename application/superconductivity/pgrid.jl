using QuantumStatistics
using StaticArrays

#
# generate a p grid for integration. dense on k and kf.
#

struct PGrid{T<:AbstractFloat, SIZE}
    k::T
    kF::T
    iscombined::Int
    grid::MVector{SIZE,T}
    size::Int
    head::T
    tail::T
    unilogs::SVector{4,Grid.UniLog{T}}

    function PGrid{T,SIZE}(k, kF, maxk, mink, MS) where {T<:AbstractFloat, SIZE}
        size = sum(MS)+1

        grid, segment,segindex = [],[],[]
        unilogs = []
        iscombined = 0

        if abs(k-kF)<2mink
            k1 = (k+kF)/2.0
            iscombined = 2
            M1 = MS[1]+MS[2]-1
            M2 = MS[3]+MS[4]-1

            g1 = Grid.UniLog{T}([0.0, k1], 1, mink, M1, 1, false, [false, true])
            push!(unilogs,g1)
            push!(unilogs,g1)
            for idx=g1.idx[1]:g1.idx[2]
                push!(grid, Grid._grid(g1, idx))
            end

            g2 = Grid.UniLog{T}([k1, maxk], M1+1, mink, M2, 1, true, [false, false])
            push!(unilogs,g2)
            push!(unilogs,g2)
            for idx=g2.idx[1]:g2.idx[2]
                push!(grid, Grid._grid(g2, idx))
            end

        elseif abs(k)<2mink
            iscombined = 1

            g0 = Grid.UniLog{T}([0.0, kF/2.0], 1, mink, MS[1]+MS[2]-1, 1, true, [false, true])
            push!(unilogs,g0)
            push!(unilogs,g0)
            for idx=g0.idx[1]:g0.idx[2]
                push!(grid, Grid._grid(g0, idx))
            end

            g1 = Grid.UniLog{T}([kF/2.0, kF], MS[1]+MS[2]+1, mink, MS[3]-1, 1, false, [false, true])
            push!(unilogs,g1)
            for idx=g1.idx[1]:g1.idx[2]
                push!(grid, Grid._grid(g1, idx))
            end

            g2 = Grid.UniLog{T}([kF, maxk], MS[1]+MS[2]+MS[3]+1, mink, MS[4]-1, 1, true, [false, false])
            push!(unilogs,g2)
            for idx=g2.idx[1]:g2.idx[2]
                push!(grid, Grid._grid(g2, idx))
            end

        else
            k1, k2 = minimum([k,kF]), maximum([k,kF])
            bounds = [[0.0, k1],[k1,(k1+k2)/2.0], [(k1+k2)/2.0, k2],[k2, maxk]]
            d2s = [false, true, false, true]
            isopen = [[false, true],[false, true],[false, true],[false, false]]
            init = 1

            for i in 1:4
                g = Grid.UniLog{T}(bounds[i], init, mink, MS[i]-1, 1, d2s[i], isopen[i])
                push!(unilogs, g)
                for idx=g.idx[1]:g.idx[2]
                    push!(grid, Grid._grid(g, idx))
                end
                init += MS[i]
            end

        end

        head, tail = grid[1], grid[end]
        return new{T,size}(k, kF, iscombined, grid, size, head, tail, unilogs)
    end

end


"""
   function pGrid(k, kF, maxk, mink, MS)

Create a logarithmic p Grid, which is densest near the Fermi momentum ``k_F`` and a given momentum ``k``

#Arguments:
- k: a given momentum
- kF: Fermi momentum
- maxK: the upper bound of the grid
- mink: the minimum interval of the grid
- MS: a list of # of points in each segments, the total size will be sum(MS)+1
"""
@inline function pGrid(k, kF, maxk, mink, MS)
    return PGrid{Float64, sum(MS)+1}(k, kF, maxk, mink, MS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pgrid = pGrid(0.001, 1.0, 10.0, 0.001, [4,4,4,4])
    println(pgrid.grid)
    pgrid = pGrid(0.999, 1.0, 10.0, 0.001, [4,4,4,4])
    println(pgrid.grid)
    pgrid = pGrid(2.0, 1.0, 10.0, 0.001, [4,4,4,4])
    println(pgrid.grid)
end
