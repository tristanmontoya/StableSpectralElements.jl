module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, GFlops, StaticArrays, Octavian
    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, GenericTensorProductAlgorithm, DefaultOperatorAlgorithm, make_operator, count_ops
    
    abstract type AbstractOperatorAlgorithm end
    struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end
    struct BLASAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericTensorProductAlgorithm <: AbstractOperatorAlgorithm end

    # default fallbacks
    make_operator(map::LinearMap, 
        ::AbstractOperatorAlgorithm) = map
    make_operator(matrix::Matrix, 
        ::AbstractOperatorAlgorithm) = OctavianMap(matrix)

    # BLAS algorithm
    make_operator(matrix::Matrix, 
        ::BLASAlgorithm) = LinearMaps.WrappedMap(matrix)
    make_operator(map::LinearMaps.UniformScalingMap,
        ::BLASAlgorithm) = map
    make_operator(map::UniformScaling, 
        ::BLASAlgorithm) = LinearMap(map,size(map,1))
    make_operator(map::LinearMap, 
        ::BLASAlgorithm) = LinearMaps.WrappedMap(Matrix(map))

    # Hand-coded matrix algorithm
    make_operator(matrix::Matrix, 
        ::GenericMatrixAlgorithm) = GenericMatrixMap(matrix)
    make_operator(map::LinearMaps.UniformScalingMap,
        ::GenericMatrixAlgorithm) = map
    make_operator(map::UniformScaling, 
        ::GenericMatrixAlgorithm) = LinearMap(map,size(map,1))
    make_operator(map::LinearMap, 
        ::GenericMatrixAlgorithm) = GenericMatrixMap(map)

    # Hand-coded Kronecker products
    function make_operator(
        map::LinearMaps.BlockMap, alg::GenericTensorProductAlgorithm)
        return vcat([make_operator(block, alg) for block in map.maps]...)
    end

    function make_operator(
        map::LinearMaps.KroneckerMap{Float64, <:NTuple{2,LinearMap}}, ::GenericTensorProductAlgorithm)
        return TensorProductMap2D(map.maps[1],map.maps[2])
    end

    function make_operator(
        map::LinearMaps.KroneckerMap{Float64, <:NTuple{3,LinearMap}}, ::GenericTensorProductAlgorithm)
        return TensorProductMap3D(map.maps[1],map.maps[2],map.maps[3])
    end
    
    # count adds, muls, and muladds
    function count_ops(map::LinearMap)
        x = rand(size(map,2))
        y = rand(size(map,1))
        cnt = @count_ops mul!(y,map,x)
        return cnt.muladd64 + cnt.add64 + cnt.mul64
    end

    export TensorProductMap2D, TensorProductMap3D
    include("tensor_product_2d.jl")
    include("tensor_product_3d.jl")

    export WarpedTensorProductMap2D, WarpedTensorProductMap3D
    include("warped_product_2d.jl")
    include("warped_product_3d.jl")

    export SelectionMap
    include("selection.jl")

    export GenericMatrixMap
    include("generic.jl")

    export OctavianMap
    include("octavian.jl")

    export ZeroMap
    include("zero.jl")

end