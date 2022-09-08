push!(LOAD_PATH,"../src/")
using Documenter, CLOUD

makedocs(sitename="CLOUD.jl")

deploydocs(
    repo = "github.com/USER_NAME/PACKAGE_NAME.jl.git",
)

