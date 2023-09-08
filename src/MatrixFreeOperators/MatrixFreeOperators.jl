module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, GFlops, StaticArrays, Octavian, TimerOutputs
    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, GenericTensorProductAlgorithm, DefaultOperatorAlgorithm, SelectionMap, make_operator, count_ops
    
    abstract type AbstractOperatorAlgorithm end
    struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end
    struct BLASAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericTensorProductAlgorithm <: AbstractOperatorAlgorithm end

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

    # union of all types that don't actually do any floating-point operations
    const NoOp = Union{SelectionMap, ZeroMap, LinearMaps.UniformScalingMap,
        LinearMaps.TransposeMap{Float64, <:SelectionMap}}

    # default fallbacks
    make_operator(matrix::Diagonal, # specialize diagonal matrices
        ::DefaultOperatorAlgorithm) = LinearMap(matrix)
    make_operator(matrix::AbstractMatrix, # gemm/gemv using Octavian.jl
        ::AbstractOperatorAlgorithm) = OctavianMap(Matrix(matrix))
    make_operator(map::UniformScaling, # specialize scalar/identity matrices
        ::AbstractOperatorAlgorithm) = LinearMap(map,size(map,1))
    make_operator(map::LinearMap, # keep predefined maps as is
        ::AbstractOperatorAlgorithm) = map

    # BLAS algorithm (OpenBLAS)
    make_operator(matrix::AbstractMatrix, 
        ::BLASAlgorithm) = LinearMaps.WrappedMap(matrix)
    make_operator(map::NoOp, ::BLASAlgorithm) = map
    make_operator(map::LinearMap, 
        ::BLASAlgorithm) = LinearMaps.WrappedMap(Matrix(map))

    # Hand-coded matrix-vector multiplication
    make_operator(matrix::Diagonal, 
        ::GenericMatrixAlgorithm) = LinearMap(matrix)
    make_operator(matrix::AbstractMatrix,
        ::GenericMatrixAlgorithm) = GenericMatrixMap(Matrix(matrix))
    make_operator(map::NoOp, ::GenericMatrixAlgorithm) = map
    make_operator(map::LinearMap, 
        ::GenericMatrixAlgorithm) = GenericMatrixMap(map)

    # Hand-coded Kronecker products
    make_operator(map::LinearMaps.BlockMap, 
        alg::GenericTensorProductAlgorithm) = vcat(
        [make_operator(block, alg) for block in map.maps]...)
    make_operator(map::LinearMaps.KroneckerMap{Float64, <:NTuple{2,LinearMap}}, 
        ::GenericTensorProductAlgorithm) = TensorProductMap2D(map.maps...)
    make_operator(map::LinearMaps.KroneckerMap{Float64, <:NTuple{3,LinearMap}},
        ::GenericTensorProductAlgorithm) = TensorProductMap3D(map.maps...)

    # count adds, muls, and muladds
    function count_ops(map::LinearMap)
        x = rand(size(map,2))
        y = rand(size(map,1))
        cnt = @count_ops mul!(y,map,x)
        return cnt.muladd64 + cnt.add64 + cnt.mul64
    end
end