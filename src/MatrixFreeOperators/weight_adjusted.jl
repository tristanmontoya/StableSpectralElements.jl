struct WeightAdjustedMap <: LinearMaps.LinearMap{Float64}
    V::LinearMap
    W::Diagonal
    Jinv::Diagonal
end

function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    M::WeightAdjustedMap,
    x::AbstractVector{Float64})
    @unpack V, W, Jinv = M
    mul!(y, inv(V' * W * Jinv * V),  x)
end

function LinearAlgebra.ldiv!(M::WeightAdjustedMap,
    rhs::AbstractArray{Float64})
    @unpack V, W, Jinv = M
    y = similar(rhs)
    mul!(y, V' * W * Jinv * V, rhs)
    rhs[:] = y
end
