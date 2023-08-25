struct TensorProductMap2D{A_type,B_type,σᵢ_type,σₒ_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    σᵢ::σᵢ_type
    σₒ::σₒ_type
end

@inline Base.size(L::TensorProductMap2D) = (size(L.σₒ,1)*size(L.σₒ,2), 
    size(L.σᵢ,1)*size(L.σᵢ,2))

function TensorProductMap2D(A, B)
    (M1,N1) = size(A)
    (M2,N2) = size(B)
    σₒ = SMatrix{M1,M2,Int}([M2*(α1-1) + α2 for α1 in 1:M1, α2 in 1:M2])
    σᵢ = SMatrix{N1,N2,Int}([N2*(β1-1) + β2 for β1 in 1:N1, β2 in 1:N2])

    if A isa LinearMaps.UniformScalingMap{Bool} 
        A = I 
    elseif A isa LinearMaps.WrappedMap
        A = SMatrix{M1,N1,Float64}(A.lmap)
    end
    if B isa LinearMaps.UniformScalingMap{Bool} 
        B = I 
    elseif B isa LinearMaps.WrappedMap
        B = SMatrix{M2,N2,Float64}(B.lmap)
    end
    return TensorProductMap2D(A,B, σᵢ, σₒ)
end

"""
Return the transpose using

(A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ 
"""
function LinearAlgebra.transpose(L::TensorProductMap2D)
    return TensorProductMap2D(transpose(L.A), transpose(L.B), L.σₒ, L.σᵢ)
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x N1*N2 matrix

L[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] B[α2,β2]

or L = A ⊗ B. The action of this matrix on a vector x is 

(Lx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] B[α2,β2] x[σᵢ[β1,β2]] 
                = ∑_{β1} A[α1,β1] (∑_{β2} B[α2,β2] x[σᵢ[β1,β2]]) 
                = ∑_{β1} A[α1,β1] Z[α2,β1] 
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap2D{<:AbstractMatrix{Float64},<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, σᵢ, σₒ) = L

    Z = MMatrix{size(σᵢ,1), size(σₒ,2),Float64}(undef)

    @inbounds for α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[β1,β2]]
        end
        Z[β1,α2] = temp
    end

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * Z[β1,α2]
        end
        y[σₒ[α1,α2]] = temp
    end

    return y
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x N1*M2 matrix

L[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] δ_{α2,β2}

or L = A ⊗ I_{M2}. The action of this matrix on a vector x is 

(Lx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] δ_{α2,β2} x[σᵢ[β1,β2]] 
                = ∑_{β1} A[α1,β1] x[σᵢ[β1,α2]] 
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap2D{<:AbstractMatrix{Float64},<:UniformScaling},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, σᵢ, σₒ) = L

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * x[σᵢ[β1,α2]]
        end
        y[σₒ[α1,α2]] = temp
    end

    return lmul!(B,y)
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x M1*N2 matrix
L[σₒ[α1,α2], σᵢ[β1,β2]] = δ_{α1,β1} B[α2,β2]

or L = I_{M1} ⊗ B. The action of this matrix on a vector x is 

(Lx)[σₒ[α1,α2]] = ∑_{β1,β2} δ_{α1,β1} B[α2,β2] x[σᵢ[β1,β2]] 
                = ∑_{β2} B[α2,β2] x[σᵢ[α1,β2]]) 
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap2D{<:UniformScaling,<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, σᵢ, σₒ) = L

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[α1,β2]]
        end
        y[σₒ[α1,α2]] = temp
    end

    return lmul!(A,y)
end