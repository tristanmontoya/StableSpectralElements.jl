struct ZeroMap <: LinearMaps.LinearMap{Float64}
    M::Int
    N::Int
end

@inline Base.size(::ZeroMap) = (M,N)

function LinearAlgebra.transpose(L::ZeroMap)
    return ZeroMap(L.N,L.M)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::ZeroMap, x::AbstractVector{Float64})
    LinearMaps.check_dim_mul(y, L, x)
    fill!(y,0.0)
    return y
end