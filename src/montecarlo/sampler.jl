"""
    createIdx!(newIdx::Int, size::Int, rng=GLOBAL_RNG)

Propose to generate new index (uniformly) randomly in [1, size]

# Arguments
- `newIdx`:  index ∈ [1, size]
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline function create!(d::Discrete, idx::Int, rng=RNG)
    d[idx] = rand(rng, d.lower:d.upper)
    return Float64(d.upper - d.lower + 1) # lower:upper has upper-lower+1 elements!
end

"""
    removeIdx!(newIdx::Int, size::Int, rng=GLOBAL_RNG)

Propose to remove the old index in [1, size]

# Arguments
- `oldIdx`:  index ∈ [1, size]
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline remove(d::Discrete, idx::Int, rng=RNG) = 1.0 / Float64(d.upper - d.lower + 1)

"""
    shiftIdx!(oldIdx::Int, newIdx::Int, size::Int, rng=GLOBAL_RNG)

Propose to shift the old index in [1, size] to a new index

# Arguments
- `oldIdx`:  old index ∈ [1, size]
- `newIdx`:  new index ∈ [1, size], will be modified!
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline function shift!(d::Discrete, idx::Int, rng=RNG)
    d[idx] = rand(rng, d.lower:d.upper)
    return 1.0
end

"""
    create!(K::FermiK{D}, idx::Int, rng=GLOBAL_RNG)

Propose to generate new Fermi K in [Kf-δK, Kf+δK)

# Arguments
- `newK`:  vector of dimension of d=2 or 3
"""
function create!(K::FermiK{D}, idx::Int, rng=RNG) where {D}
    ############ Simple Way ########################
    # for i in 1:DIM
    #     newK[i] = Kf * (rand(rng) - 0.5) * 2.0
    # end
    # return (2.0 * Kf)^DIM
    ################################################

    Kamp = K.kF + (rand(rng) - 0.5) * 2.0 * K.δk
    (Kamp <= 0.0) && return 0.0
    # Kf-dK<Kamp<Kf+dK 
    ϕ = 2π * rand(rng)
    if D == 3 # dimension 3
        θ = π * rand(rng)
        # newK .= Kamp .* Mom(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ))
        K[idx] = @SVector [Kamp * cos(ϕ) * sin(θ), Kamp * sin(ϕ) * sin(θ), Kamp * cos(θ)]
        return 2 * K.δk * 2π * π * (sin(θ) * Kamp^2)
        # prop density of KAmp in [Kf-dK, Kf+dK), prop density of Phi
        # prop density of Theta, Jacobian
    else  # DIM==2
        K[idx] = @SVector [Kamp * cos(ϕ), Kamp * sin(ϕ)]
        return 2 * K.δk * 2π * Kamp
        # prop density of KAmp in [Kf-dK, Kf+dK), prop density of Phi, Jacobian
    end
end

"""
    removeFermiK!(oldK, Kf=1.0, δK=0.5, rng=GLOBAL_RNG)

Propose to remove an existing Fermi K in [Kf-δK, Kf+δK)

# Arguments
- `oldK`:  vector of dimension of d=2 or 3
"""
function remove(K::FermiK{D}, idx::Int, rng=RNG) where {D}
    ############## Simple Way #########################
    # for i in 1:DIM
    #     if abs(oldK[i]) > Kf
    #         return 0.0
    #     end
    # end
    # return 1.0 / (2.0 * Kf)^DIM
    ####################################################

    oldK = K[idx]
    Kamp = sqrt(dot(oldK, oldK))
    if !(K.kF - K.δk < Kamp < K.kF + K.δk)
        return 0.0
    end
    # (Kamp < Kf - dK || Kamp > Kf + dK) && return 0.0
    if D == 3 # dimension 3
        sinθ = sqrt(oldK[1]^2 + oldK[2]^2) / Kamp
        sinθ < 1.0e-15 && return 0.0
        return 1.0 / (2 * K.δk * 2π * π * sinθ * Kamp^2)
    else  # DIM==2
        return 1.0 / (2 * K.δk * 2π * Kamp)
end
end

"""
    shiftK!(oldK, newK, step, rng=GLOBAL_RNG)

Propose to shift oldK to newK. Work for generic momentum vector
"""
function shift!(K::FermiK{D}, idx::Int, rng=RNG) where {D}
    x = rand(rng)
    if x < 1.0 / 3
        # K[idx] = K[idx] + (rand(rng, D) .- 0.5) .* K.δk
        K[idx] += (rand(rng, D) .- 0.5) .* K.δk
        return 1.0
    elseif x < 2.0 / 3
        λ = 1.5
        ratio = 1.0 / λ + rand(rng) * (λ - 1.0 / λ)
        K[idx] = K[idx] * ratio
        return (D == 2) ? 1.0 : ratio
    else
        K[idx] = K[idx] * (-1.0)
        return 1.0
    end
    # x = rand(rng)
    # K = K[idx]
    # if x < 0.5
    #     λ = 1.5
    #     ratio = 1.0 / λ + rand(rng) * (λ - 1.0 / λ)
    #     K = K * ratio
    #     return (D == 2) ? 1.0 : ratio
    # else
    #     ϕ = rand(rng) * 2π
    #     if (D == 3)
    #         θ = rand(rng) * π
    #         if (θ == 0.0)
    #             return 0.0
    #         end
    #         Kamp = sqrt(K[1]^2 + K[2]^2 + K[3]^2)
    #         K = [Kamp * cos(ϕ) * sin(θ), Kamp * sin(ϕ) * sin(θ), Kamp * cos(θ)]
    #         # return 2π^2/(Kamp^2*sin(θ))
    #         return 1.0 / 2π^2 * (Kamp^2 * sin(θ))
    #     else # D=2
    #         Kamp = sqrt(K[1]^2 + K[2]^2)
    #         K = [Kamp * cos(ϕ), Kamp * sin(ϕ)]
    #         return 1.0
    #     end
    # end
end

"""
    create!(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to generate new tau (uniformly) randomly in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.t[idx] will be updated
"""
@inline function create!(T::Tau, idx::Int, rng=RNG)
    T[idx] = rand(rng) * T.β
    return T.β
end

"""
    remove(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to remove old tau in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.t[idx] will be updated
"""
@inline function remove(T::Tau, idx::Int, rng=RNG)
    return 1.0 / T.β
end

"""
    shift!(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to shift the old tau to new tau, both in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.t[idx] will be updated
"""
@inline function shift!(T::Tau, idx::Int, rng=RNG)
    x = rand(rng)
    if x < 1.0 / 3
        T[idx] = T[idx] + 2 * T.λ * (rand(rng) - 0.5)
    elseif x < 2.0 / 3
        T[idx] = T.β - T[idx]
    else
        T[idx] = rand(rng) * T.β
    end

    if T[idx] < 0.0
        T[idx] += T.β
    elseif T[idx] > T.β
        T[idx] -= T.β
    end

    return 1.0
end


"""
    create!(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to generate new angle (uniformly) randomly in [0, 2π), return proposal probability

# Arguments
- `theta`:  angle variable
- `idx`: theta.t[idx] will be updated
"""
@inline function create!(theta::Angle, idx::Int, rng=RNG)
    theta[idx] = rand(rng) * 2π
    return 2π
end

"""
    remove(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to remove old theta in [0, 2π), return proposal probability

# Arguments
    - `theta`:  Tau variable
- `idx`: theta.t[idx] will be updated
"""
@inline function remove(theta::Angle, idx::Int, rng=RNG)
    return 1.0 / 2.0 /π
end

"""
    shift!(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to shift the old theta to new theta, both in [0, 2π), return proposal probability

# Arguments
- `theta`:  angle variable
- `idx`: theta.t[idx] will be updated
"""
@inline function shift!(theta::Angle, idx::Int, rng=RNG)
    x = rand(rng)
    if x < 1.0 / 3
        theta[idx] = theta[idx] + 2 * theta.λ * (rand(rng) - 0.5)
    elseif x < 2.0 / 3
        theta[idx] = π - theta[idx]
    else
        theta[idx] = rand(rng) * 2π
    end

    if theta[idx] < 0.0
        theta[idx] += 2π
    elseif theta[idx] > 2π
        theta[idx] -= 2π
    end

    return 1.0
end
