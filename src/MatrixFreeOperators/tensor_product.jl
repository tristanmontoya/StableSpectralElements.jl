struct TensorProductMap{A_type,B_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    σᵢ::Matrix{Int}
    σₒ::Matrix{Int}
end

@inline Base.size(C::TensorProductMap) = (size(C.σₒ,1)*size(C.σₒ,2), 
    size(C.σᵢ,1)*size(C.σᵢ,2))

"""
Compute transpose using
(A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ 
"""
function LinearAlgebra.transpose(C::TensorProductMap)
    return TensorProductMap(transpose(C.A), transpose(C.B), C.σₒ, C.σᵢ)
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
    C::TensorProductMap{<:AbstractMatrix{Float64},<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C
    (M1,M2) = size(σₒ)
    (N1,N2) = size(σᵢ)

    Z = Matrix{Float64}(undef, M2, N1)
    @turbo for α2 in 1:M2, β1 in 1:N1
        Zij = 0.0
        for β2 in 1:N2
            @muladd Zij = Zij + B[α2,β2]*x[σᵢ[β1,β2]]
        end
        Z[α2,β1] = Zij
    end

    @turbo for α1 in 1:M1, α2 in 1:M2
        yi = 0.0
        for β1 in 1:N1
            @muladd yi = yi + A[α1,β1]*Z[α2,β1]
        end
        y[σₒ[α1,α2]] = yi
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
    C::TensorProductMap{<:AbstractMatrix{Float64},<:UniformScaling},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C
    (M1,M2) = size(σₒ)
    (N1,N2) = size(σᵢ)

    @turbo for α1 in 1:M1, α2 in 1:M2
        yi = 0.0
        for β1 in 1:N1
            @muladd yi = yi + A[α1,β1]*x[σᵢ[β1,α2]]
        end
        y[σₒ[α1,α2]] = yi
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
    C::TensorProductMap{<:UniformScaling,<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C
    (M1,M2) = size(σₒ)
    (N1,N2) = size(σᵢ)

    @turbo for α1 in 1:M1, α2 in 1:M2
        yi = 0.0
        for β2 in 1:N2
            @muladd yi = yi + B[α2,β2]*x[σᵢ[α1,β2]]
        end
        y[σₒ[α1,α2]] = yi
    end

    y = A * y
    return y
end
