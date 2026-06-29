module MatrixFreeOperators

using LinearAlgebra, LinearMaps, MuladdMacro, StaticArrays, Octavian, TimerOutputs
export AbstractOperatorAlgorithm,
       BLASAlgorithm,
       GenericMatrixAlgorithm,
       GenericTensorProductAlgorithm,
       DefaultOperatorAlgorithm,
       SelectionMap,
       make_operator

abstract type AbstractOperatorAlgorithm end
struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end
struct BLASAlgorithm <: AbstractOperatorAlgorithm end
struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
struct GenericTensorProductAlgorithm <: AbstractOperatorAlgorithm end

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

export SplitSimplexMap
include("split_simplex.jl")

# union of all types that don't actually do any floating-point operations
const NoOp = Union{SelectionMap,
    ZeroMap,
    LinearMaps.UniformScalingMap,
    LinearMaps.TransposeMap{Float64, <:SelectionMap}}

# default fallbacks
make_operator(matrix::Diagonal, # specialize diagonal matrices
::DefaultOperatorAlgorithm) = LinearMap(matrix)
function make_operator(matrix::AbstractMatrix, # gemm/gemv using Octavian.jl
        ::AbstractOperatorAlgorithm)
    OctavianMap(Matrix(matrix))
end
function make_operator(map::UniformScaling, # specialize scalar/identity matrices
        ::AbstractOperatorAlgorithm)
    LinearMap(map, size(map, 1))
end
make_operator(map::LinearMap, # keep predefined maps as is
::AbstractOperatorAlgorithm) = map

# BLAS algorithm (OpenBLAS)
make_operator(matrix::AbstractMatrix, ::BLASAlgorithm) = LinearMaps.WrappedMap(matrix)
make_operator(map::NoOp, ::BLASAlgorithm) = map
make_operator(map::LinearMap, ::BLASAlgorithm) = LinearMaps.WrappedMap(Matrix(map))

# Hand-coded matrix-vector multiplication
make_operator(matrix::Diagonal, ::GenericMatrixAlgorithm) = LinearMap(matrix)
function make_operator(matrix::AbstractMatrix, ::GenericMatrixAlgorithm)
    GenericMatrixMap(Matrix(matrix))
end
make_operator(map::NoOp, ::GenericMatrixAlgorithm) = map
make_operator(map::LinearMap, ::GenericMatrixAlgorithm) = GenericMatrixMap(map)

# Hand-coded Kronecker products
function make_operator(map::LinearMaps.BlockMap, alg::GenericTensorProductAlgorithm)
    vcat([make_operator(block, alg) for block in map.maps]...)
end
function make_operator(map::LinearMaps.KroneckerMap{Float64, <:NTuple{2, LinearMap}},
        ::GenericTensorProductAlgorithm)
    make_operator(map.maps[1], GenericMatrixAlgorithm()) ⊗
    make_operator(map.maps[2], GenericMatrixAlgorithm())
end
function make_operator(map::LinearMaps.KroneckerMap{Float64, <:NTuple{3, LinearMap}},
        ::GenericTensorProductAlgorithm)
    make_operator(map.maps[1], GenericMatrixAlgorithm()) ⊗
    make_operator(map.maps[2], GenericMatrixAlgorithm()) ⊗
    make_operator(map.maps[3], GenericMatrixAlgorithm())
end

end
