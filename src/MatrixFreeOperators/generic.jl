struct GenericMatrixMap <: LinearMaps.LinearMap{Float64}
    lmap::Matrix{Float64}
end

function GenericMatrixMap(L::LinearMap)
    return GenericMatrixMap(Matrix(L))
end

@inline Base.size(L::GenericMatrixMap) = size(L.lmap)
function LinearAlgebra.transpose(L::GenericMatrixMap)
    return GenericMatrixMap(transpose(L.lmap))
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64},
        L::GenericMatrixMap,
        x::AbstractVector{Float64})
    LinearMaps.check_dim_mul(y, L, x)
    @inbounds for i in eachindex(y)
        temp = 0.0
        @inbounds @simd for j in eachindex(x)
            @muladd temp = temp + L.lmap[i, j] * x[j]
        end
        y[i] = temp
    end

    return y
end
