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

function vertexDynamic(config, qd, qe, τIn, τOut)
    vd, wd = interactionDynamic(config, qd, τIn, τOut)
    ve, we = interactionDynamic(config, qe, τIn, τOut)

    return -vd, -wd, ve, we
end