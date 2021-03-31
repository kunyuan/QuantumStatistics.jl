var documenterSearchIndex = {"docs":
[{"location":"lib/utility/#Utility","page":"Utility","title":"Utility","text":"","category":"section"},{"location":"lib/utility/","page":"Utility","title":"Utility","text":"Modules = [QuantumStatistics.Utility]","category":"page"},{"location":"lib/utility/#QuantumStatistics.Utility","page":"Utility","title":"QuantumStatistics.Utility","text":"Utility data structures and functions\n\n\n\n\n\n","category":"module"},{"location":"lib/utility/#QuantumStatistics.Utility.StopWatch","page":"Utility","title":"QuantumStatistics.Utility.StopWatch","text":"StopWatch(start, interval, callback)\n\nInitialize a stopwatch. \n\nArguments\n\nstart::Float64: initial time (in seconds)\ninterval::Float64 : interval to click (in seconds)\ncallback : callback function after each click (interval seconds)\n\n\n\n\n\n","category":"type"},{"location":"lib/utility/#QuantumStatistics.Utility.check-Tuple{QuantumStatistics.Utility.StopWatch,Vararg{Any,N} where N}","page":"Utility","title":"QuantumStatistics.Utility.check","text":"check(stopwatch, parameter...)\n\nCheck stopwatch. If it clicks, call the callback function with the unpacked parameter\n\n\n\n\n\n","category":"method"},{"location":"lib/twopoint/#Two-point-correlators","page":"Two-point correlators","title":"Two-point correlators","text":"","category":"section"},{"location":"lib/twopoint/","page":"Two-point correlators","title":"Two-point correlators","text":"Modules = [QuantumStatistics.TwoPoint]","category":"page"},{"location":"lib/twopoint/#QuantumStatistics.TwoPoint","page":"Two-point correlators","title":"QuantumStatistics.TwoPoint","text":"Provide N-body response and correlation functions\n\n\n\n\n\n","category":"module"},{"location":"lib/twopoint/#QuantumStatistics.TwoPoint.fermiT-Union{Tuple{T}, Tuple{T,T}, Tuple{T,T,T}} where T<:AbstractFloat","page":"Two-point correlators","title":"QuantumStatistics.TwoPoint.fermiT","text":"fermiT(τ, ϵ, β = 1.0)\n\nCompute the bare fermionic Green's function. Assume k_B=hbar=1\n\ng(τ0) = e^-ϵτ(1+e^-βϵ) g(τ0) = -e^-ϵτ(1+e^βϵ)\n\nArguments\n\nτ: the imaginary time, must be (-β, β]\nϵ: dispersion minus chemical potential: E_k-μ      it could also be the real frequency ω if the bare Green's function is used as the kernel in the Lehmann representation \nβ = 1.0: the inverse temperature \n\n\n\n\n\n","category":"method"},{"location":"lib/twopoint/#QuantumStatistics.TwoPoint.fermiΩ-Union{Tuple{T}, Tuple{T,Int64,T}} where T<:AbstractFloat","page":"Two-point correlators","title":"QuantumStatistics.TwoPoint.fermiΩ","text":"bareFermiMatsubara(β, n, ε, [, scale])\n\nCompute the bare Green's function for a given Matsubara frequency.\n\ng(iω_n) = -1(iω_n-ε)\n\nwhere ω_n=(2n+1)πβ. The convention here is consist with the book \"Quantum Many-particle Systems\" by J. Negele and H. Orland, Page 95\n\nArguments\n\nβ: the inverse temperature \nτ: the imaginary time, must be (-β, β]\nε: dispersion minus chemical potential: E_k-μ;       it could also be the real frequency ω if the bare Green's function is used as the kernel in the Lehmann representation \n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#Monte-Carlo","page":"Monte Carlo","title":"Monte Carlo","text":"","category":"section"},{"location":"lib/montecarlo/","page":"Monte Carlo","title":"Monte Carlo","text":"Modules = [QuantumStatistics.MonteCarlo]","category":"page"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo","text":"Monte Carlo Calculator for Diagrams\n\n\n\n\n\n","category":"module"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.Diagram","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.Diagram","text":"Group{A}(type::Int, internal::Tuple{Vararg{Int}}, external::Tuple{Vararg{Int}}, eval, obstype=Float64)\n\ncreate a group of diagrams\n\n#Arguments:\n\ntype: integer identifier of the group\ninternal: internal variable numbers, e.g. [number of internal momentum, number of internal tau]\nexternal: array of size of external index, e.g. [size of external momentum index, size of external tau]\neval: function to evaluate the group\nobstype: type of the diagram weight, e.g. Float64\n\n\n\n\n\n","category":"type"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.create!","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.create!","text":"createIdx!(newIdx::Int, size::Int, rng=GLOBAL_RNG)\n\nPropose to generate new index (uniformly) randomly in [1, size]\n\nArguments\n\nnewIdx:  index ∈ [1, size]\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.create!-2","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.create!","text":"create!(T::Tau, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate new tau (uniformly) randomly in [0, β), return proposal probability\n\nArguments\n\nT:  Tau variable\nidx: T.t[idx] will be updated\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.create!-Union{Tuple{D}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64,Any}} where D","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.create!","text":"create!(K::FermiK{D}, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate new Fermi K in [Kf-δK, Kf+δK)\n\nArguments\n\nnewK:  vector of dimension of d=2 or 3\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.progressBar-Tuple{Any,Any}","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.progressBar","text":"progressBar(step, total)\n\nReturn string of progressBar (step/total*100%)\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.remove","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.remove","text":"removeIdx!(newIdx::Int, size::Int, rng=GLOBAL_RNG)\n\nPropose to remove the old index in [1, size]\n\nArguments\n\noldIdx:  index ∈ [1, size]\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.remove-2","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.remove","text":"remove(T::Tau, idx::Int, rng=GLOBAL_RNG)\n\nPropose to remove old tau in [0, β), return proposal probability\n\nArguments\n\nT:  Tau variable\nidx: T.t[idx] will be updated\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.remove-Union{Tuple{D}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64,Any}} where D","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.remove","text":"removeFermiK!(oldK, Kf=1.0, δK=0.5, rng=GLOBAL_RNG)\n\nPropose to remove an existing Fermi K in [Kf-δK, Kf+δK)\n\nArguments\n\noldK:  vector of dimension of d=2 or 3\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.shift!","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.shift!","text":"shiftIdx!(oldIdx::Int, newIdx::Int, size::Int, rng=GLOBAL_RNG)\n\nPropose to shift the old index in [1, size] to a new index\n\nArguments\n\noldIdx:  old index ∈ [1, size]\nnewIdx:  new index ∈ [1, size], will be modified!\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.shift!-2","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.shift!","text":"shift!(T::Tau, idx::Int, rng=GLOBAL_RNG)\n\nPropose to shift the old tau to new tau, both in [0, β), return proposal probability\n\nArguments\n\nT:  Tau variable\nidx: T.t[idx] will be updated\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#QuantumStatistics.MonteCarlo.shift!-Union{Tuple{D}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64}, Tuple{QuantumStatistics.MonteCarlo.FermiK{D},Int64,Any}} where D","page":"Monte Carlo","title":"QuantumStatistics.MonteCarlo.shift!","text":"shiftK!(oldK, newK, step, rng=GLOBAL_RNG)\n\nPropose to shift oldK to newK. Work for generic momentum vector\n\n\n\n\n\n","category":"method"},{"location":"lib/spectral/#Spectral-functions","page":"Spectral functions","title":"Spectral functions","text":"","category":"section"},{"location":"lib/spectral/","page":"Spectral functions","title":"Spectral functions","text":"Modules = [QuantumStatistics.Spectral]","category":"page"},{"location":"lib/spectral/#QuantumStatistics.Spectral","page":"Spectral functions","title":"QuantumStatistics.Spectral","text":"Spectral representation related functions\n\n\n\n\n\n","category":"module"},{"location":"lib/spectral/#QuantumStatistics.Spectral.fermiDirac-Union{Tuple{T}, Tuple{T}} where T<:AbstractFloat","page":"Spectral functions","title":"QuantumStatistics.Spectral.fermiDirac","text":"fermiDirac(ω)\n\nCompute the Fermi Dirac function. Assume k_B Thbar=1\n\nf(ω) = 1(1+e^-ω)\n\nArguments\n\nω: frequency\n\n\n\n\n\n","category":"method"},{"location":"lib/spectral/#QuantumStatistics.Spectral.kernelFermiT-Union{Tuple{T}, Tuple{T,T}} where T<:AbstractFloat","page":"Spectral functions","title":"QuantumStatistics.Spectral.kernelFermiT","text":"kernelFermiT(τ, ω)\n\nCompute the imaginary-time fermionic kernel. Assume k_B Thbar=1\n\ng(τ0) = e^-ωτ(1+e^-ω) g(τ0) = -e^-ωτ(1+e^ω)\n\nArguments\n\nτ: the imaginary time, must be (-1, 1]\nω: frequency\n\n\n\n\n\n","category":"method"},{"location":"lib/diagram/#Simple-Diagrams","page":"Simple Diagrams","title":"Simple Diagrams","text":"","category":"section"},{"location":"lib/diagram/","page":"Simple Diagrams","title":"Simple Diagrams","text":"Modules = [QuantumStatistics.Diagram]","category":"page"},{"location":"lib/diagram/#QuantumStatistics.Diagram","page":"Simple Diagrams","title":"QuantumStatistics.Diagram","text":"Calculator for some simple diagrams\n\n\n\n\n\n","category":"module"},{"location":"lib/diagram/#QuantumStatistics.Diagram.bubble-Union{Tuple{T}, Tuple{T,Complex{T},Int64}, Tuple{T,Complex{T},Int64,Any}, Tuple{T,Complex{T},Int64,Any,Any}, Tuple{T,Complex{T},Int64,Any,Any,Any}, Tuple{T,Complex{T},Int64,Any,Any,Any,Any}} where T<:AbstractFloat","page":"Simple Diagrams","title":"QuantumStatistics.Diagram.bubble","text":"bubble(q, ω, dim, kF, β)\n\nCompute the polarization function of free electrons at a given frequency. \n\nArguments\n\nq: external momentum, q<1e-4 will be treated as q=0 \nω: externel frequency, make sure Im ω>0\ndim: dimension\nkF=1.0: Fermi momentum \nβ=1.0: inverse temperature\nm=1/2: mass\n`dispersion': dispersion, default k^2/2m-kF^2/2m\neps=1.0e-6: the required absolute accuracy\n\n\n\n\n\n","category":"method"},{"location":"#QuantumStatistics.jl","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"","category":"section"},{"location":"","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"A toolbox for quantum many-body field theory.","category":"page"},{"location":"#Outline","page":"QuantumStatistics.jl","title":"Outline","text":"","category":"section"},{"location":"","page":"QuantumStatistics.jl","title":"QuantumStatistics.jl","text":"Pages = [\n    \"lib/grid.md\",\n    \"lib/spectral.md\",\n    \"lib/green.md\",\n    \"lib/twopoint.md\",\n    \"lib/montecarlo.md\",\n    \"lib/fastmath.md\",\n    \"lib/utility.md\",\n]\nDepth = 1","category":"page"},{"location":"lib/fastmath/#Fast-Math-Functions","page":"Fast Math Functions","title":"Fast Math Functions","text":"","category":"section"},{"location":"lib/fastmath/","page":"Fast Math Functions","title":"Fast Math Functions","text":"Modules = [QuantumStatistics.FastMath]","category":"page"},{"location":"lib/fastmath/#QuantumStatistics.FastMath","page":"Fast Math Functions","title":"QuantumStatistics.FastMath","text":"Provide a set of fast math functions\n\n\n\n\n\n","category":"module"},{"location":"lib/fastmath/#QuantumStatistics.FastMath.invsqrt-Tuple{Float64}","page":"Fast Math Functions","title":"QuantumStatistics.FastMath.invsqrt","text":"invsqrt(x)\n\nThe Legendary Fast Inverse Square Root See the following links: wikipedia and thesis\n\n\n\n\n\n","category":"method"},{"location":"lib/grid/#Grids","page":"Grids","title":"Grids","text":"","category":"section"},{"location":"lib/grid/","page":"Grids","title":"Grids","text":"Modules = [QuantumStatistics.Grid]","category":"page"},{"location":"lib/grid/#QuantumStatistics.Grid.Uniform","page":"Grids","title":"QuantumStatistics.Grid.Uniform","text":"Uniform{Type,SIZE}\n\nCreate a uniform Grid with a given type and size\n\nMember:\n\nβ: inverse temperature\nhalfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)\nsize: the Grid size\ngrid: vector stores the grid\nsize: size of the grid vector\nhead: grid head\ntail: grid tail\nδ: distance between two grid elements\nisopen: if isopen[1]==true, then grid[1] will be slightly larger than the grid head. Same for the tail.\n\n\n\n\n\n","category":"type"},{"location":"lib/grid/#QuantumStatistics.Grid.boseK","page":"Grids","title":"QuantumStatistics.Grid.boseK","text":"boseK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), twokFi = floor(Int, 2size / 3), type = Float64)\n\nCreate a logarithmic bosonic K Grid, which is densest near the momentum 0 and 2k_F\n\n#Arguments:\n\nKf: Fermi momentum\nmaxK: the upper bound of the grid\nhalfLife: the grid is densest in the range (0, Kf+halfLife) and (2Kf-halfLife, 2Kf+halfLife)\nsize: the Grid size\nkFi: index of Kf\ntwokFi: index of 2Kf\n\n\n\n\n\n","category":"function"},{"location":"lib/grid/#QuantumStatistics.Grid.fermiK","page":"Grids","title":"QuantumStatistics.Grid.fermiK","text":"fermiK(Kf, maxK, halfLife, size::Int, kFi = floor(Int, 0.5size), type = Float64)\n\nCreate a logarithmic fermionic K Grid, which is densest near the Fermi momentum k_F\n\n#Arguments:\n\nKf: Fermi momentum\nmaxK: the upper bound of the grid\nhalfLife: the grid is densest in the range (Kf-halfLife, Kf+halfLife)\nsize: the Grid size\nkFi: index of Kf\n\n\n\n\n\n","category":"function"},{"location":"lib/grid/#QuantumStatistics.Grid.linear2D-NTuple{5,Any}","page":"Grids","title":"QuantumStatistics.Grid.linear2D","text":"linear2D(xgrid, ygrid, data, x, y) \n\nlinear interpolation of data(x, y)\n\n#Arguments:\n\nxgrid: one-dimensional grid of x\nygrid: one-dimensional grid of y\ndata: two-dimensional array of data\nx: x\ny: y\n\n\n\n\n\n","category":"method"},{"location":"lib/grid/#QuantumStatistics.Grid.tau","page":"Grids","title":"QuantumStatistics.Grid.tau","text":"tau(β, halfLife, size::Int, type = Float64)\n\nCreate a logarithmic Grid for the imaginary time, which is densest near the 0 and β\n\n#Arguments:\n\nβ: inverse temperature\nhalfLife: the grid is densest in the range (0, halfLife) and (β-halfLife, β)\nsize: the Grid size\n\n\n\n\n\n","category":"function"}]
}
