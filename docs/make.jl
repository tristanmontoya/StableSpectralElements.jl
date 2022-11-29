import Pkg
Pkg.add("Documenter")
using Documenter
using CLOUD

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "CLOUD.jl",
    authors = "Tristan Montoya",
    pages = [
        "Home" => "index.md",
        "`ConservationLaws`" => "ConservationLaws.md",
        "`SpatialDiscretizations`" => "SpatialDiscretizations.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/favicon.ico"],
        ansicolor=true
    )
)
deploydocs(
    repo="github.com/tristanmontoya/CLOUD.jl.git"
)