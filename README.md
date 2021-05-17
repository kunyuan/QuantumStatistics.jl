# Package QuantumStatistics

Toolbox for numerical experiment on quantum many-body field theory.

|                                                                   Status                                                                    |                                                                      Coverage                                                                      |                                                     Document                                                      |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| [![Build Status](https://travis-ci.org/kunyuan/QuantumStatistics.jl.svg?branch=master)](https://travis-ci.org/kunyuan/QuantumStatistics.jl) | [![codecov](https://codecov.io/gh/kunyuan/QuantumStatistics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kunyuan/QuantumStatistics.jl) | [![Document](https://img.shields.io/badge/docs-dev-blue.svg)](https://kunyuan.github.io/QuantumStatistics.jl/dev) |

## Documentation
<!-- - [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.** -->
- [**DEVEL**](https://kunyuan.github.io/QuantumStatistics.jl/dev/) &mdash; *documentation of the in-development version.*

## Features

The package is in development stage. Many components are missing. For now, we have implemented the following utilities:

- Compact Lehmann representations for imaginary-time/Matsubara frequency Green's function.
- A general purpose Monte Carlo integrator to calculate Feynman diagrams.
- Grids to properly handle the discretized imaginary-time and momentum.
<!-- - Fast elementary math functions. Some of them are adapted from the package [Yeppp.jl](https://github.com/JuliaMath/Yeppp.jl). It supports more generic array types than the original package. -->


## Installation

Currently, this package is not yet registered. So, `Pkg.add("QuantumStatistics")` will not work (yet).

There two ways to install this package:

1. The package can be installed with the Julia package manager. 
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/kunyuan/QuantumStatistics.jl
```

2. Alternatively, you can run the following command from the Julia REPL:
```julia
julia> using Pkg; Pkg.add(PackageSpec(url="https://github.com/kunyuan/QuantumStatistics.jl"))
```

## Development

1. To develop or modify this package, you need to install this package first, then type `]` to enter the Pkg REPL mode and run:
```julia
pkg> dev QuantumStatistics
```
This command will automatically create a copy of the package git repository in at ~/.julia/dev/QuantumStatistics.

2. To check if the dev package is correctly activated,  you may enter the Pkg REPL mode and run,
```julia
pkg> st
```
you should be able to see the QuantumStatistics entry points to the above folder.

3. You may make modifications to the dev package. When you load the package from somewhere,
```julia
using QuantumStatistics
```
you should be able to see the modifications.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an issue if you encounter any problems.

<!-- Example of Julia package to go along with [these notes](https://tlienart.github.io/pub/julia/dev-pkg2.html). -->
<!-- https://travis-ci.org/github/kunyuan/QuantumStatistics.jl -->
<!-- [![codecov](https://codecov.io/gh/kunyuan/QuantumStatistics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kunyuan/QuantumStatistics.jl) -->
