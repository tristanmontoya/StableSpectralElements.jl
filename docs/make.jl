push!(LOAD_PATH,"../src/")
using Documenter

makedocs(sitename="CLOUD.jl")

deploydocs(#
    repo="github.com/tristanmontoya/CLOUD.jl.git"
)