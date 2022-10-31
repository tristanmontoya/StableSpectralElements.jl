# In progress...
struct TensorProductMap3D{A_type,B_type,C_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    C::C_type
    σᵢ::Array{Int,3}
    σₒ::Array{Int,3}
end

@inline Base.size(map::TensorProductMap3D) = (
    size(map.σₒ,1)*size(map.σₒ,2)*size(map.σₒ,3), 
    size(map.σᵢ,1)*size(map.σᵢ,2)*size(map.σᵢ,3))

"""
Compute transpose using
(A ⊗ B ⊗ C)ᵀ = Aᵀ ⊗ Bᵀ ⊗ Cᵀ
"""
function LinearAlgebra.transpose(map::TensorProductMap3D)
    return TensorProductMap3D(transpose(map.A), transpose(map.B), 
        transpose(map.C), map.σₒ, map.σᵢ)
end