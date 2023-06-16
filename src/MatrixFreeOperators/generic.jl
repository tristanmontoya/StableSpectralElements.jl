struct GenericMatrixMap <: LinearMaps.LinearMap{Float64}
    A::AbstractMatrix
end

function GenericMatrixMap(L::LinearMap)
    return GenericMatrixMap(Matrix(L))
end

@inline Base.size(L::GenericMatrixMap) = size(L.A)
function LinearAlgebra.transpose(L::GenericMatrixMap)
    return GenericMatrixMap(transpose(L.A))
end

function LinearAlgebra.mul!(y::AbstractVector{Float64},
    L::GenericMatrixMap, x::AbstractVector{Float64})
    (; A) = L

    for i in eachindex(y)
        temp = 0.0
        @simd for j in eachindex(x)
            @muladd temp = temp + A[i,j]*x[j]
        end
        y[i] = temp
    end
    
    return y
end