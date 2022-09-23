module MatrixFreeOperators

    using LinearAlgebra, LinearMaps
    using UnPack
    
    const Operator1D{T} = Union{UniformScaling{Bool}, 
        Matrix{T}, Transpose{T,Matrix{T}}}
    
    export TensorProductMap
    include("tensor_product.jl")

    export WarpedTensorProductMap
    include("warped_product.jl")

    export SelectionMap
    include("selection.jl")
end