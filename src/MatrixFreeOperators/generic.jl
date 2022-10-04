struct GenericMatrixMap <: LinearMaps.LinearMap{Float64}
    A::AbstractMatrix
end

function GenericMatrixMap(map::LinearMap)
    return GenericMatrixMap(Matrix(map))
end

@inline Base.size(map::GenericMatrixMap) = size(map.A)

function LinearAlgebra.transpose(map::GenericMatrixMap)
    return GenericMatrixMap(transpose(map.A))
end

function LinearAlgebra.mul!(y::AbstractVector{Float64},
    map::GenericMatrixMap, x::AbstractVector{Float64})
    @unpack A = map

    @turbo for i in eachindex(y)
        temp = 0.0
        for j in eachindex(x)
            @muladd temp = temp + A[i,j]*x[j]
        end
        y[i] = temp
    end
    
    return y
end