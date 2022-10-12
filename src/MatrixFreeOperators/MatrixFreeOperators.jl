module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, UnPack, LoopVectorization

    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, DefaultOperatorAlgorithm, combine, make_operator
    
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

    function combine(map::LinearMap)
        return LinearMap(convert(Matrix,map))
    end
    
    function LinearAlgebra.mul!(y::AbstractVector, 
        A::Diagonal, x::AbstractVector)
        y[:] = A.diag .* x
        return y
    end

    export TensorProductMap
    include("tensor_product.jl")

    export WarpedTensorProductMap
    include("warped_product.jl")

    export SelectionMap
    include("selection.jl")

    export GenericMatrixMap
    include("generic.jl")

end