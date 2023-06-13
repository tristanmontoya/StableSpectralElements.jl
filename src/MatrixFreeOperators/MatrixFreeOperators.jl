module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, UnPack, GFlops, StaticArrays
    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, GenericTensorProductAlgorithm, DefaultOperatorAlgorithm, make_operator, count_ops
    
    abstract type AbstractOperatorAlgorithm end
    struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end
    struct BLASAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericTensorProductAlgorithm <: AbstractOperatorAlgorithm end

    function make_operator(map::LinearMap, ::DefaultOperatorAlgorithm)
        return map
    end
    
    function make_operator(matrix::Matrix, ::DefaultOperatorAlgorithm)
        return LinearMaps.WrappedMap(matrix)
    end

    function make_operator(matrix::Matrix, ::BLASAlgorithm)
        return LinearMaps.WrappedMap(matrix)
    end

    function make_operator(map::LinearMap, ::BLASAlgorithm)
        return LinearMaps.WrappedMap(Matrix(map))
    end

    function make_operator(matrix::Matrix, ::GenericMatrixAlgorithm)
        return GenericMatrixMap(matrix)
    end

    function make_operator(map::LinearMap, ::GenericMatrixAlgorithm)
        return GenericMatrixMap(map)
    end

    function make_operator(
        map::LinearMaps.KroneckerMap{Float64, <:NTuple{2,LinearMap}}, ::GenericTensorProductAlgorithm)
        return TensorProductMap2D(map.maps[1],map.maps[2])
    end

    function make_operator(
        map::LinearMaps.KroneckerMap{Float64, <:NTuple{3,LinearMap}}, ::GenericTensorProductAlgorithm)
        return TensorProductMap3D(map.maps[1],map.maps[2],map.maps[3])
    end

    function make_operator(map::LinearMap, ::GenericTensorProductAlgorithm)
        return map
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

    export ZeroMap
    include("zero.jl")

end