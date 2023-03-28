module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, UnPack, GFlops

    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, DefaultOperatorAlgorithm, make_operator, count_ops
    
    abstract type AbstractOperatorAlgorithm end
    struct BLASAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
    struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end

    function make_operator(map::LinearMap, ::DefaultOperatorAlgorithm)
        return map
    end
    
    function make_operator(map::LinearMap, ::BLASAlgorithm)
        return LinearMaps.WrappedMap(Matrix(map))
    end

    function make_operator(map::LinearMap, ::GenericMatrixAlgorithm)
        return GenericMatrixMap(map)
    end

    function count_ops(map::LinearMap)
        x = rand(size(map,2))
        y = rand(size(map,1))
        cnt = @count_ops mul!(y,map,x)
        return cnt.muladd64 + cnt.add64 + cnt.mul64
    end

    export TensorProductMap2D, TensorProductMap3D
    include("tensor_product_2d.jl")
    include("tensor_product_3d.jl")

    export WarpedTensorProductMap2D
    include("warped_product_2d.jl")

    export WarpedTensorProductMap3D
    include("warped_product_3d.jl")

    export SelectionMap
    include("selection.jl")

    export GenericMatrixMap
    include("generic.jl")

    export WeightAdjustedMap
    include("weight_adjusted.jl")

end