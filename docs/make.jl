import Pkg
Pkg.add("Documenter")
using Documenter
using CLOUD

push!(LOAD_PATH,"../src/")

makedocs(sitename="CLOUD.jl", authors = "Tristan Montoya")

deploydocs(#
    repo="github.com/tristanmontoya/CLOUD.jl.git"
)