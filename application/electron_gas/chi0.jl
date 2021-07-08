using QuantumStatistics, Parameters, Printf
using Roots, Polylogarithms
using Plots
# using PackageCompiler
# --sysimage sys_plots.so
# create_sysimage(:Plots, sysimage_path="sys_plots.so", precompile_execution_file="precompile_plots.jl")
include("parameter.jl")
# gr()
# Plots.GRBackend()

@with_kw struct Para
    n::Int = 0 # external Matsubara frequency
    Qsize::Int = 17
    Tsize::Int = 16
    extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0, 32.0 * kF, Qsize)]
    extT::Vector{SVector{3,Float64}} = [@SVector [T, 0.0, 0.0] for T in LinRange(10, 610, Tsize)]
    fixQ::Float64 = 10* kF
end

function chemical_potential(beta)
    f(β,μ)=real(polylog(3/2,-exp(β*μ)))+4/3/π^0.5*(β)^(3/2)
    g(μ)=f(beta,μ)
    return find_zero(g, (-10000,1))
end

function chi0vsq(beta)
    @unpack n, Qsize, extQ = Para()

    μ=chemical_potential(beta)*EF
    println("μ=$μ, reduced β=$beta")

    pdata = Matrix{Float64}(undef, Qsize, 2)
    for (idx, q) in enumerate(extQ)
        q = q[1]
        p, err = TwoPoint.LindhardΩnFiniteTemperature(dim, q, n, μ, kF, β, me, spin)
        @printf("%10.6f  %10.6f ± %10.6f\n", q / kF, p, err)
        pdata[idx,1] = q/kF
        # pdata[idx,1] = (kF/q)^2
        pdata[idx,2] = -p
        # pdata[idx,1] = (kF/q)^4 
        pdata[idx,2] = -p-2/3/π^2*EF^(3/2)/q^2
    end
    return pdata
end

function chi0vsT()
    @unpack n, Tsize, extT, fixQ = Para()
    pdata = Matrix{Float64}(undef, Tsize, 2)
    for (idx, T) in enumerate(extT)
        T = T[1]
        μ=chemical_potential(1.0/T)*EF
        println("μ=$μ, reduced T=$T, fixed extMom q=$fixQ")
        β_in = 1.0/T/kF^2

        p, err = TwoPoint.LindhardΩnFiniteTemperature(dim, fixQ, n, μ, kF, β_in, me, spin)
        @printf("%10.6f  %10.6f ± %10.6f\n", T, p, err)
        # pdata[idx,1] = T
        pdata[idx,1] = 1.0/T
        pdata[idx,2] = -p
    end
    return pdata, fixQ
end
# labels = [string(pdata[i,1]) for i in 1:Tsize]


# @unpack Tsize, extT = Para()
# for (idx, T) in enumerate(extT)
#     T = T[1]
#     pdata = chi0vsq(1.0/T)
#     plot!(pdata[:,1], pdata[:,2], xlabel="q/k_F", ylabel="χ_0-(1/q^2 term)", label="T=$T")
#     # plt = plot(pdata[:,1], pdata[:,2], xlim=(0, 0.1), ylim=(0,) xlabel="(k_F/q)^2", ylabel="χ_0", label="β=$beta", title = "ideal polarization")
#     # plt = plot(pdata[:,1], pdata[:,2], xlabel="(k_F/q)^4", ylabel="χ_0-(1/q^2 term)", label="β=$beta", title = "ideal polarization")
# end
# savefig("chi0vsq_T_v1.pdf")
# show(plt)

pdata, fixQ = chi0vsT()
qint = round(Int, fixQ/kF) 
# plt = plot(pdata[:,1], pdata[:,2], xlim=(0,0.2), ylim=(0,0.0017), xlabel="√β", ylabel="χ_0(q=$qint k_F)", title = "ideal polarization", legend=:none)
plt = plot(pdata[:,1], pdata[:,2], xlim=(0,0.04), ylim=(0,0.0017), xlabel="β", ylabel="χ_0(q=$qint k_F)", title = "ideal polarization", legend=:none)
savefig("chi0vsβ_q$qint.pdf")
display(plt)

