# In progress...
struct TensorProductMap3D{A_type,B_type,C_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    C::C_type
    σᵢ::Array{Int,3}
    σₒ::Array{Int,3}
end

@inline Base.size(L::TensorProductMap3D) = (
    size(L.σₒ,1)*size(L.σₒ,2)*size(L.σₒ,3), 
    size(L.σᵢ,1)*size(L.σᵢ,2)*size(L.σᵢ,3))

"""
Compute transpose using
(A ⊗ B ⊗ C)ᵀ = Aᵀ ⊗ Bᵀ ⊗ Cᵀ
"""
function LinearAlgebra.transpose(L::TensorProductMap3D)
    return TensorProductMap3D(transpose(L.A), transpose(L.B), 
        transpose(L.C), L.σₒ, L.σᵢ)
end