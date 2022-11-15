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
        "Modules" => "modules.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)
deploydocs(
    repo="github.com/tristanmontoya/CLOUD.jl.git"
)