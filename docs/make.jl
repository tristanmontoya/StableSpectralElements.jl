using Documenter
using StableSpectralElements

makedocs(
    sitename = "StableSpectralElements.jl",
    authors = "Tristan Montoya",
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "`ConservationLaws`" => "ConservationLaws.md",
            "`SpatialDiscretizations`" => "SpatialDiscretizations.md",
            "`Solvers`" => "Solvers.md",
        ],
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/favicon.ico"],
        ansicolor = true,
    ),
)
deploydocs(repo = "github.com/tristanmontoya/StableSpectralElements.jl.git")
