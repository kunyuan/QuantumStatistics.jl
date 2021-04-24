
function Sphere1(totalstep, pid)
    obs1 = 0.0
    obs2 = 0.0

    function integrand(config)
        if config.curr.id == 1
            return 1.0
        else
            X = config.var[1]
            if X[1]^2 + X[2]^2 < 1.0
                return 1.0
            else
                return 0.0
            end
        end
    end

    function measure(config)
        diag = config.curr
        factor = 1.0 / config.absWeight / diag.reWeightFactor
        if diag.id == 1
            obs1 += factor
        elseif diag.id == 2
            weight = integrand(config)
            obs2 += weight * factor
        else
            error("Not implemented!")
        end
    end

    rng = MersenneTwister(pid)

    T = MonteCarlo.Tau(1.0, 1.0 / 2.0)
    diag1 = MonteCarlo.Diagram(1, 0, [0,]) # id, order, [T num, ]
    diag2 = MonteCarlo.Diagram(2, 1, [2,]) # id, order, [T num, ]

    config = MonteCarlo.Configuration(totalstep, (diag1, diag2), (T,); pid=pid, rng=rng)
    MonteCarlo.montecarlo(config, (x) -> abs(integrand(x)), measure, print=false)

    return obs2 / obs1
end

function Sphere2(totalstep, pid)
    obs1 = 0.0
    obs2 = 0.0

    function integrand(config)
        if config.curr.id == 1
            return 1.0
        else
            X = config.var[1]
            if X[1][1]^2 + X[1][2]^2 < 1.0
                return 1.0
            else
                return 0.0
            end
        end
    end

    function measure(config)
        diag = config.curr
        factor = 1.0 / config.absWeight / diag.reWeightFactor
        if diag.id == 1
            obs1 += factor
        elseif diag.id == 2
            weight = integrand(config)
            obs2 += weight * factor
        else
            error("Not implemented!")
        end
    end

    rng = MersenneTwister(pid)

    T = MonteCarlo.TauPair(1.0, 1.0 / 2.0)
    diag1 = MonteCarlo.Diagram(1, 0, [0,]) # id, order, [T num, ]
    diag2 = MonteCarlo.Diagram(2, 1, [1,]) # id, order, [T num, ]

    config = MonteCarlo.Configuration(totalstep, (diag1, diag2), (T,); pid=pid, rng=rng)
    MonteCarlo.montecarlo(config, integrand, measure, print=false)

    return obs2 / obs1
end

@testset "MonteCarlo Sampler" begin
    repeat = 64
    totalStep = 100000

    observable = map((x) -> Sphere1(totalStep, rand(1:100000)), 1:repeat)

    obs = mean(observable)
    err = std(observable) / sqrt(length(observable))

    println("MC integration 1: $obs ± $err (exact: $(π / 4.0))")
    @test abs(obs - π / 4.0) < 5.0 * err

    observable = map((x) -> Sphere2(totalStep, rand(1:100000)), 1:repeat)

    obs = mean(observable)
    err = std(observable) / sqrt(length(observable))

    println("MC integration 2: $obs ± $err (exact: $(π / 4.0))")
    @test abs(obs - π / 4.0) < 5.0 * err
end
