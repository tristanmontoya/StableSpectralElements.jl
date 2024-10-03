struct OctavianMap <: LinearMaps.LinearMap{Float64}
    lmap::Matrix{Float64}
end

@inline function OctavianMap(L::LinearMap)
    return OctavianMap(Matrix(L))
end

@inline Base.size(L::OctavianMap) = size(L.lmap)

function LinearAlgebra.transpose(L::OctavianMap)
    return OctavianMap(transpose(L.lmap))
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64},
        L::OctavianMap,
        x::AbstractVector{Float64})
    LinearMaps.check_dim_mul(y, L, x)
    matmul_serial!(y, L.lmap, x)
    return y
end
