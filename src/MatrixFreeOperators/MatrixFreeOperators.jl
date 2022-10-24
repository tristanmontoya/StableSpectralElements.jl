module MatrixFreeOperators

    using LinearAlgebra, LinearMaps, MuladdMacro, UnPack

    export AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, DefaultOperatorAlgorithm, combine, make_operator, count_ops
    
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
    
    function count_ops(map::LinearMap)
        x = rand(size(map,2))
        y = rand(size(map,1))
        cnt = @count_ops mul!(y,map,x)
        return cnt.muladd64 + cnt.add64 + cnt.mul64
    end

    ```
    Remove unnecessary add operations in diagonal matrix multiplication
    See Julia issue 47312 at https://github.com/JuliaLang/julia/issues/47312
    ```
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