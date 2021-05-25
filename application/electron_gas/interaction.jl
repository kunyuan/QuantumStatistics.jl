# global constant e0 and mass2 is expected
function interactionDynamic(config, qd, τIn, τOut)
    para = config.para

    dτ = abs(τOut - τIn)

    kDiQ = sqrt(dot(qd, qd))
    vd = 4π * e0^2 / (kDiQ^2 + mass2)
    if kDiQ <= para.qgrid.grid[1]
        q = para.qgrid.grid[1] + 1.0e-6
        wd = vd * Grid.linear2D(para.dW0, para.qgrid, para.τgrid, q, dτ)
        # the current interpolation vanishes at q=0, which needs to be corrected!
    else
        wd = vd * Grid.linear2D(para.dW0, para.qgrid, para.τgrid, kDiQ, dτ) # dynamic interaction, don't forget the singular factor vq
    end

    return vd / β, wd
end

# fake dynamic interaction for benchmark purpose
function WRPA(τ, q)
    # temporarily using a toy model
    g = 1.0

    factor = 1.0
    q2 = dot(q, q)
    if τ < -β
        τ = τ + 2 * β
    elseif τ < 0
        τ = τ + β
        factor = 1.0
    end
    sq2g = sqrt(q2 + g)
    return -factor * 4 * π / (q2 + g) / sq2g * (exp(-sq2g * τ) + exp(-sq2g * (β - τ))) / (1 - exp(-sq2g * β))
end

function vertexDynamic(config, qd, qe, τIn, τOut)
    vd, wd = interactionDynamic(config, qd, τIn, τOut)
    ve, we = interactionDynamic(config, qe, τIn, τOut)

    # wd = WRPA(τOut - τIn, qd)
    # we = WRPA(τOut - τIn, qe)

    return -vd, -wd, ve, we
end