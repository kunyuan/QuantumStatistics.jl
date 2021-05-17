# push!(LOAD_PATH, "../src/")
using Documenter, QuantumStatistics

makedocs(
    modules=[QuantumStatistics], 
    sitename="QuantumStatistics.jl",
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "Monte Carlo integrator" => "man/important_sampling.md",
        ],
        "Library" => Any[
            map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
            # "Internals" => map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
        ]
    ]

)

deploydocs(repo="github.com/kunyuan/QuantumStatistics.jl.git")
