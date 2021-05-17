# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using QuantumStatistics, LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters, StaticArrays
# using ProfileView
using PyCall

special = pyimport("scipy.special")
const l = 9
coeff = SVector{l+1,Float64}(special.legendre(l).c)
exp = zeros(l+1)
for i=1:l+1
    exp[i]=l+1-i
end
exp = SVector{l+1,Float64}(exp)

function legendre(x)
    return dot((x .^ exp) ,coeff)
end

function test()
    x = 1.3
    y1 = legendre(x)
    y2 = special.legendre(l)(x)
    @printf("%10.6f , %10.6f\n", y1, y2)
end

test()
const Steps = 1e7

const e0 = sqrt(2)  # electric charge
const me = 0.5  # electron mass
const dim = 3    # dimension (D=2 or 3, doesn't work for other D!!!)
const spin = 2  # number of spins

const rs = 2.0  
const kF =1.0# (dim == 3) ? (9π / (2spin))^(1 / 3) / rs : sqrt(4 / spin) / rs
const EF = kF^2 / (2me)
const β = 1000.0 / kF^2
const mass2 = 0.01

const e2 = rs * 2.0/(9π/4.0)^(1.0/3)

# function P_(z,u)
#     return 1.0+(1.0-z^2+u^2)/4/z*log(((1+z)^2+u^2)/((1-z)^2+u^2)) - u *atan(2*u/(u^2+z^2-1))
# end

function P_(z::Float64)
    return 1.0+(1.0-z^2)/2.0/z*log(((1+z))/(abs(1-z)))
end

# function Π(q,ω)
#     return me * kF/2/π^2 * P_(q/2.0/kF, me*ω/q/kF)
# end

function Π0(q::Float64)
    return me * kF/2/π^2 * P_(q/2.0/kF)
end

function integrand(config)
    if config.curr != 1
        error("impossible")
    end

    T = config.var[1]
    z = 2kF * T[1]

    k, p = kF, kF

    W = 1.0/( z^2/(4π*e2) + Π0(z))

    return 2kF * z/(k*p) * legendre((k^2+p^2-z^2)/2/k/p) * W
end

function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    obs[1] += weight / abs(weight) * factor
end

function run(steps)

    T = MonteCarlo.Tau(1.0, 1.0 / 2.0)

    dof = [[1,],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64, 1) # observable for the normalization diagram and the bubble

    config = MonteCarlo.Configuration(steps, (T,), dof, obs)
    avg, std = MonteCarlo.sample(config, integrand, measure; print=0, Nblock=16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    if isnothing(avg) == false
        @printf("%10.6f ± %10.6f\n", avg[1], std[1])
    end
end

run(Steps)
#@time run(Steps)
