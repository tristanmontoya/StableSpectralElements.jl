struct TensorProductMap2D{A_type,B_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    σᵢ::Matrix{Int}
    σₒ::Matrix{Int}
end

@inline Base.size(C::TensorProductMap2D) = (size(C.σₒ,1)*size(C.σₒ,2), 
    size(C.σᵢ,1)*size(C.σᵢ,2))

"""
Compute transpose using
(A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ 
"""
function LinearAlgebra.transpose(C::TensorProductMap2D)
    return TensorProductMap2D(transpose(C.A), transpose(C.B), C.σₒ, C.σᵢ)
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x N1*N2 matrix

C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] B[α2,β2]

or C = A ⊗ B. The action of this matrix on a vector x is 

(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] B[α2,β2] x[σᵢ[β1,β2]] 
                = ∑_{β1} A[α1,β1] (∑_{β2} B[α2,β2] x[σᵢ[β1,β2]]) 
                = ∑_{β1} A[α1,β1] Z[α2,β1] 
"""
function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    C::TensorProductMap2D{<:AbstractMatrix{Float64},<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C

    Z = Matrix{Float64}(undef, size(σₒ,2), size(σᵢ,1))
    @inbounds for α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @inbounds @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[β1,β2]]
        end
        Z[α2,β1] = temp
    end

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @inbounds @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * Z[α2,β1]
        end
        y[σₒ[α1,α2]] = temp
    end

    return y
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x N1*M2 matrix

C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] δ_{α2,β2}

or C = A ⊗ I_{M2}. The action of this matrix on a vector x is 

(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] δ_{α2,β2} x[σᵢ[β1,β2]] 
                = ∑_{β1} A[α1,β1] x[σᵢ[β1,α2]] 
"""
function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    C::TensorProductMap2D{<:AbstractMatrix{Float64},<:UniformScaling},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C
    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @inbounds @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * x[σᵢ[β1,α2]]
        end
        y[σₒ[α1,α2]] = temp
    end

    y = B * y
    return y
end

"""
Multiply a vector of length N1*N2 by the M1*M2 x M1*N2 matrix
C[σₒ[α1,α2], σᵢ[β1,β2]] = δ_{α1,β1} B[α2,β2]

or C = I_{M1} ⊗ B. The action of this matrix on a vector x is 

(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} δ_{α1,β1} B[α2,β2] x[σᵢ[β1,β2]] 
                = ∑_{β2} B[α2,β2] x[σᵢ[α1,β2]]) 
"""
function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    C::TensorProductMap2D{<:UniformScaling,<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2)
        temp = 0.0
        @inbounds @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[α1,β2]]
        end
        y[σₒ[α1,α2]] = temp
    end

    y = A * y
    return y
end
