module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, UnPack
    
    export TensorProductMap
    include("tensor_product.jl")

    export WarpedTensorProductMap
    include("warped_product.jl")

    export SelectionMap
    include("selection.jl")
end