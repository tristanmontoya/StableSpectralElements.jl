using Pkg
pkg"activate .."
using Documenter, CLOUD

makedocs(sitename="CLOUD.jl")

deploydocs(
    repo = "github.com/tristanmontoya/CLOUD.jl",
)

