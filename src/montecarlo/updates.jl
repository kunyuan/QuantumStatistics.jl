
function increaseOrder(config, integrand)
    idx = rand(config.rng, 1:length(config.diagrams))
    curr = config.curr
    new = config.diagrams[idx]
    if (new.order != curr.order + 1)
        return
    end

    prop = 1.0
    for vi in 1:length(config.var)
        for pos = curr.nvar[vi] + 1:new.nvar[vi]
            prop *= create!(config.var[vi], pos, config.rng)
        end
    end

    currAbsWeight = config.absWeight

    config.curr = new
    newAbsWeight = abs(integrand(config))
    R = prop * newAbsWeight * new.reWeightFactor / currAbsWeight / curr.reWeightFactor

    curr.propose[1] += 1.0
    if rand(config.rng) < R
        curr.accept[1] += 1.0
        config.curr = new
        config.absWeight = newAbsWeight
    else
        config.curr = curr
        # in case the user modifies config.absWeight when calculate integrand(config)
        config.absWeight = currAbsWeight
    end
end

function decreaseOrder(config, integrand)
    idx = rand(config.rng, 1:length(config.diagrams))
    new = config.diagrams[idx]
    curr = config.curr

    if (new.order != curr.order - 1)
        return
    end

    prop = 1.0
    for vi in 1:length(config.var)
        for pos = new.nvar[vi] + 1:curr.nvar[vi]
            prop *= remove(config.var[vi], pos, config.rng)
        end
    end

    config.curr = new
    currAbsWeight = config.absWeight
    newAbsWeight = abs(integrand(config))
    R = prop * newAbsWeight * new.reWeightFactor / currAbsWeight / curr.reWeightFactor
    curr.propose[2] += 1.0
    if rand(config.rng) < R
        curr.accept[2] += 1.0
        config.absWeight = newAbsWeight
        config.curr = new
    else
        # in case the user modifies config.absWeight when calculate integrand(config)
        config.absWeight = currAbsWeight
        config.curr = curr
    end
end

function changeVar(config, integrand)
    curr = config.curr
    vi = rand(config.rng, 1:length(curr.nvar))
    var = config.var[vi]
    (curr.nvar[vi] <= 0) && return # return if the var number is less than 1
    idx = rand(config.rng, 1:curr.nvar[vi]) # randomly choose one var to update
    oldvar = copy(var[idx])
    prop = shift!(var, idx, config.rng)

    currAbsWeight = config.absWeight
    newAbsWeight = abs(integrand(config))
    R = prop * newAbsWeight / currAbsWeight
    curr.propose[2 + vi] += 1.0
    if rand(config.rng) < R
        curr.accept[2 + vi] += 1.0
        config.absWeight = newAbsWeight
    else
        var[idx] = oldvar
        # in case the user modifies config.absWeight when calculate integrand(config)
        config.absWeight = currAbsWeight 
    end
end