using FastGaussQuadrature
include("chebyshev.jl")
include("pgrid.jl")

idx(ki, xi, order) = (ki - 1) * order + xi

struct CompositeGrid
    grid::Vector{Float64} # panel x finegrid
    wgrid::Vector{Float64}
    panel::Vector{Float64}
    x::Vector{Float64}
    w::Vector{Float64}
    # xp::Matrix{Float64} # the fine grid
    # wp::Matrix{Float64} # the weight of the fine grid
    Np::Int  # number of panel - 1
    order::Int  # number of fine grid
    type::Symbol
    function CompositeGrid(panel, order, type::Symbol)
        if type == :cheb
            x, w = barychebinit(order)
        elseif type == :gaussian
            x, w = gausslegendre(order)
        else
            throw("not implemented!")
        end

        # println("Quadrature type : ", type)
        # println("Quadrature points : ", x)
        # println("Quadrature weights: ", w)

        Np = length(panel) - 1

        grid = zeros(Float64, Np * order)
        wgrid = zeros(Float64, Np * order)
        # xp = zeros(Float64, (Np, order))
        # wp = zeros(Float64, (Np, order))

        for p in 1:Np
            a, b = panel[p], panel[p + 1]
            for xi in 1:order
                index = idx(p, xi, order)
                grid[index] = (a + b) / 2 + (b - a) / 2 * x[xi]
                wgrid[index] = (b - a) / 2 * w[xi]
            end
        end

        return new(grid, wgrid, panel, x, w, Np, order, type)
    end
end

function KPanel(Nk, kF, maxK, minK)
    panel = Grid.boseKUL(0.5 * kF, maxK, minK, Nk, 1).grid
    panel[1] = 0.0  # the kgrid start with 0.0
    return panel
end

function QPanel(Nk, kF, maxK, minK, k)
    panel = pGrid(k, kF, maxK, minK, [Nk,Nk,Nk,Nk]).grid

    # panel = Grid.boseKUL(0.5 * kF, maxK, minK, Nk, 1).grid
    # panel[1] = 0.0  # the kgrid start with 0.0

    return panel
end

"""
interpolate!(f, k::CompositeGrid, grid)

map f array in the grid k to a new grid 

# Arguments
- f::Vector{Float64}: vector of data.
- k::CompositeGrid: the grid object that f is defined on
- grid::Vector{Float64}: new grid points 
"""
function interpolate(f, k::CompositeGrid, grid)
    @assert k.type == :cheb
    order = k.order
    ff = zeros(eltype(f), length(grid))
    kpidx = 1 # panel index of the kgrid
    head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
    # extract all x in the kpidx-th k panel
    fx = @view f[head:tail] # all F in the same kpidx-th K panel
    x = @view k.grid[head:tail]
    w = @view k.wgrid[head:tail]

    for (qi, q) in enumerate(grid)
        # for a given q, one needs to find the k panel to do interpolation
        if q > k.panel[kpidx + 1]
            # if q is too large, move k panel to the next
            # println("before $q, $kpidx, $(k.panel[kpidx]) -> $(k.panel[kpidx + 1])")
            while q > kgrid.panel[kpidx + 1]
                kpidx += 1
            end
            # println("after $q, $kpidx, $(k.panel[kpidx]) -> $(k.panel[kpidx + 1])")
            head, tail = idx(kpidx, 1, order), idx(kpidx, order, order) 
            fx = @view f[head:tail] # all F in the same kpidx-th K panel
            x = @view k.grid[head:tail]
            w = @view k.wgrid[head:tail]
            @assert kpidx <= k.Np
        end
        ff[qi] = barycheb(order, q, fx, w, x) # the interpolation is independent with the panel length
        # @assert k.panel[kpidx - 1][order] <= q <= k.panel[pidx + 1][1] "$q for kpidx=$kpidx with $x"
    end
    return ff
end

