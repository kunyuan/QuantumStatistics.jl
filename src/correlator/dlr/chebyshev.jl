"""
barychebinit(n)

Get Chebyshev nodes of first kind and corresponding barycentric Lagrange interpolation weights. 
Reference: Berrut, J.P. and Trefethen, L.N., 2004. Barycentric lagrange interpolation. SIAM review, 46(3), pp.501-517.

# Arguments
- `n`: order of the Chebyshev interpolation

# Returns
- Chebyshev nodes
- Barycentric Lagrange interpolation weights
"""
function barychebinit(n)
    x = zeros(Float64, n)
    w = similar(x)
    for i in 1:n
        c = (2i - 1)π / (2n)
        x[n - i + 1] = cos(c)
        w[n - i + 1] = (-1)^(i - 1) * sin(c)
    end
    return x, w
end

"""
function barycheb(n, x, f, wc, xc)

Barycentric Lagrange interpolation at Chebyshev nodes
Reference: Berrut, J.P. and Trefethen, L.N., 2004. Barycentric lagrange interpolation. SIAM review, 46(3), pp.501-517.

# Arguments
- `n`: order of the Chebyshev interpolation
- `x`: coordinate to interpolate
- `f`: array of size n, function at the Chebyshev nodes
- `wc`: array of size n, Barycentric Lagrange interpolation weights
- `xc`: array of size n, coordinates of Chebyshev nodes

# Returns
- Interpolation result
"""
function barycheb(n, x, f, wc, xc)
    for j in 1:n
        if x == xc[j]
            return f[j]
        end    
    end

    num, den = 0.0, 0.0
    for j in 1:n
        q = wc[j] / (x - xc[j])
        num += q * f[j]
        den += q
    end
    return num / den
end