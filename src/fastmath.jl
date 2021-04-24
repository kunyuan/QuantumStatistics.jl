"""
Provide a set of fast math functions
"""
module FastMath
using LinearAlgebra
using Cuba
export dot, squaredNorm, norm, integrator1D

# include("Yeppp.jl")
# using .Yeppp

# @inline dot(x, y) = Yeppp.dot(x, y)
# @inline squaredNorm(x) = Yeppp.dot(x, x)
# @inline norm(x) = sqrt(Yeppp.dot(x, x))

@inline dot(x, y) = dot(x, y)
@inline squaredNorm(x) = dot(x, x)
@inline norm(x) = sqrt(dot(x, x))

"""
    invsqrt(x)

The Legendary Fast Inverse Square Root
See the following links: [wikipedia](https://en.wikipedia.org/wiki/Fast_inverse_square_root) and [thesis](https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf)
"""
@inline function invsqrt(x::Float64)
    #   y = x
    x2::Float64 = x * 0.5
    i::Int64 = reinterpret(Int64, x)
    # The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    i = 0x5fe6eb50c7b537a9 - (i >> 1)
    y = reinterpret(Float64, i)
    y = y * (1.5 - (x2 * y * y)) # 1st iteration
    y = y * (1.5 - (x2 * y * y))  # 2nd iteration, this can be removed
    return y
end

@inline function invsqrt(x::Float32)
    x2 = x * 0.5f0
    i::Int32 = reinterpret(Int32, x)  # evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1)
    y = reinterpret(Float32, i)
    y = y * (1.5f0 - (x2 * y * y)) # 1st iteration
    y = y * (1.5f0 - (x2 * y * y))  # 2nd iteration, this can be removed
    return y
end

function integrator1D(type, f, xmin, xmax, rtol)
    function integrand(x, g)
        if xmin != -Inf && xmax != Inf
            a, b = xmin, xmax
            g[1] = f(a + (b - a) * x[1]) * (b - a)
        elseif xmin != -Inf && xmax == Inf
            a = xmin
            g[1] = f(a + x[1] / (1 - x[1])) / (1 - x[1])^2
        elseif xmin == -Inf && xmax != Inf
            b = xmax
            g[1] = - f(b + 1 - 1 / x[1]) / (x[1])^2
        else
            y = x[1]
            denorm = (1 - y) * y
            g[1] = - f((2y - 1) / denorm) * (2y^2 - 2y + 1) / denorm^2
        end
    end

    if type == :vegas
        result, err = Cuba.vegas(integrand, rtol=rtol)
        return result[1], err[1]
    elseif type == :cuhre
        result, err = Cuba.cuhre(integrand, rtol=rtol)
        return result[1], err[1]
    else
        @error "Not implemented!"
    end
end

end