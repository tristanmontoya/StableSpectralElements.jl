struct ZeroMap <: LinearMaps.LinearMap{Float64}
    M::Int
    N::Int
end

@inline Base.size(L::ZeroMap) = (L.M, L.N)

function LinearAlgebra.transpose(L::ZeroMap)
    return ZeroMap(L.N, L.M)
end

function LinearAlgebra.mul!(y::AbstractVector{Float64},
        L::ZeroMap,
        x::AbstractVector{Float64})
    LinearMaps.check_dim_mul(y, L, x)
    fill!(y, 0.0)
    return y
end
