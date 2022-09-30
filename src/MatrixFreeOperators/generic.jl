struct GenericMatrixMap <: LinearMaps.LinearMap{Float64}
    A::AbstractMatrix
end

function GenericMatrixMap(C::LinearMap)
    return GenericMatrixMap(Matrix(C))
end

@inline Base.size(C::GenericMatrixMap) = size(C.A)

function LinearAlgebra.transpose(C::GenericMatrixMap)
    return GenericMatrixMap(transpose(C.A))
end

function LinearAlgebra.mul!(y::AbstractVector{Float64},
    C::GenericMatrixMap, x::AbstractVector{Float64})
    
    @turbo for i in eachindex(y)
        yi = 0.0
        for j in eachindex(x)
            @muladd yi = yi + C.A[i,j]*x[j]
        end
        y[i] = yi
    end
        
    return y
end