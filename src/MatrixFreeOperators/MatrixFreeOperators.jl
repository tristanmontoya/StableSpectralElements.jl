module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, GFlops, StaticArrays
    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, GenericTensorProductAlgorithm, DefaultOperatorAlgorithm, make_operator, count_ops, flux_difference!
    
    abstract type AbstractOperatorAlgorithm end
    struct DefaultOperatorAlgorithm <: AbstractOperatorAlgorithm end
    struct BLASAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericMatrixAlgorithm <: AbstractOperatorAlgorithm end
    struct GenericTensorProductAlgorithm <: AbstractOperatorAlgorithm end

    function make_operator(map::LinearMap, ::AbstractOperatorAlgorithm)
        return map
    end
    
    function make_operator(matrix::Matrix, ::AbstractOperatorAlgorithm)
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
    
    function count_ops(map::LinearMap)
        x = rand(size(map,2))
        y = rand(size(map,1))
        cnt = @count_ops mul!(y,map,x)
        return cnt.muladd64 + cnt.add64 + cnt.mul64
    end

    function flux_difference!(y::AbstractVector{Float64}, 
        A::AbstractMatrix{Float64}, F::AbstractMatrix{Float64})
        @assert size(A) == size(F)
        @assert size(y,1) == size(A,1)

        for i in axes(y,1)
            y[i] = dot(A[i,:], F[i,:])
        end
        
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

    export ZeroMap
    include("zero.jl")

end