"""
Monte Carlo Calculator for Diagrams
"""
module MonteCarlo

export montecarlo, Configuration, Diagram, FermiK, BoseK, Tau, TauPair

using Random
using LinearAlgebra
using StaticArrays, Printf, Dates
using ..Utility
const RNG = Random.GLOBAL_RNG

include("variable.jl")
include("sampler.jl")
include("updates.jl")

function montecarlo(config::Configuration, integrand::Function, measure::Function; timer=[], print=true)
    ##############  initialization  ################################

    # don't forget to initialize the diagram weight
    config.absWeight = abs(integrand(config))

    if print
        printTime = 10
        push!(timer, StopWatch(printTime, printStatus))
    end

    updates = [increaseOrder, decreaseOrder]
    for var in config.var
        # changeVar should be call more often if there are more variables
        push!(updates, changeVar)
    end

    for diag in config.diagrams
        diag.propose = zeros(Float64, length(updates)) .+ 1.0e-8
        # add a small number so that the ratio propose/accept=0 if there is no such update proposed and accepted
        diag.accept = zeros(Float64, length(updates)) 
    end

    ########### MC simulation ##################################
    if (print)
        printstyled("PID $(config.pid) Start Simulation ...\n", color=:red)
    end
    startTime = time()

    for i = 1:config.totalStep
        config.step += 1
        config.curr.visitedSteps += 1
        _update = rand(config.rng, updates) # randomly select an update
        _update(config, integrand)
        (i % 10 == 0 && i >= config.totalStep / 100) && measure(config)
        if i % 1000 == 0
            for t in timer
                check(t, config)
            end
            if i > 1000_1000 && i % 1000_1000 == 0
                reweight(config)
            end
        end
    end

    if (print)
        printStatus(config)
        printstyled("PID $(config.pid) End Simulation. Cost $(time() - startTime) seconds.\n\n", color=:red)
    end
end

function reweight(config)
    # config.diagrams[1].reWeightFactor = 1.0
    # config.diagrams[2].reWeightFactor = 8.0
    avgstep = sum([g.visitedSteps for g in config.diagrams]) / length(config.diagrams)
    for g in config.diagrams
        if g.visitedSteps > 10000
            # g.reWeightFactor=g.reWeightFactor*0.5+totalstep/g.visitedSteps*0.5
            g.reWeightFactor *= avgstep / g.visitedSteps
        end
    end
end

function printStatus(config)
    barbar = "====================================================================================="
    bar = "-------------------------------------------------------------------------------------"

    println(barbar)
    printstyled(Dates.now(), color=:green)
    println("\nStep:", config.step)
    println(bar)

    name = ["increaseOrder", "decreaseOrder"]
    for (vi, var) in enumerate(config.var)
        # typeof(Var) is something like QuantumStatistics.MonteCarlo.Tau, only the last block is the type name
        typestr = "$(typeof(var))"
        typestr = split(typestr, ".")[end]
        append!(name, ["change_$typestr", ])
    end
    totalproposed = 0.0

    for num = 1:length(name)
        @printf(
            "%-20s %12s %12s %12s\n",
            String(name[num]),
            "Proposed",
            "Accepted",
            "Ratio  "
        )
        for (idx, diag) in enumerate(config.diagrams)
            @printf(
                "    Order%2d:            %11.6f%% %11.6f%% %12.6f\n",
                diag.id,
                diag.propose[num] / config.step * 100.0,
                diag.accept[num] / config.step * 100.0,
                diag.accept[num] / diag.propose[num]
            )
            totalproposed += diag.propose[num]
        end
        println(bar)
    end
    printstyled("Diagrams            Visited      ReWeight\n", color=:yellow)
    for (idx, diag) in enumerate(config.diagrams)
        @printf(
                "  Order%2d:     %12i %12.6f\n",
                diag.id,
                diag.visitedSteps,
                diag.reWeightFactor
            )
    end
    println(bar)
    printstyled("Total Proposed: $(totalproposed / config.step * 100.0)%\n", color=:yellow)
    # printstyled(progressBar(round(config.step / config.blockStep, digits=2), config.totalBlock + 1), color=:green)
    printstyled(progressBar(config.step, config.totalStep), color=:green)
    println()
end

"""
    progressBar(step, total)

Return string of progressBar (step/total*100%)
"""
function progressBar(step, total)
    barWidth = 70
    percent = round(step / total * 100.0, digits=2)
    str = "["
    pos = barWidth * percent / 100.0
    for i = 1:barWidth
        if i <= pos
            str *= "█"
        else
            str *= " "
        end
    end
    str *= "] $step/$total=$percent%"
    return str
end

end
