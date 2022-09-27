"""
Warped tensor-product operator (e.g. for Dubiner-type bases)
"""    
struct WarpedTensorProductMap{A_type, B_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::Vector{B_type}
    σᵢ::Matrix{Int}
    σₒ::Matrix{Int}
end

Base.size(C::WarpedTensorProductMap) = (count(a->a>0,C.σₒ), 
    count(a->a>0,C.σᵢ))

"""
Multiply a vector of length Σ_{β1}N2[β1] by the M1*M2 x Σ_{β1}N2[β1] matrix

C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] B[β1][α2,β2].

The action of this matrix on a vector x is 

(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] B[β1][α2,β2] x[σᵢ[β1,β2]] 
                = ∑_{β1} A[α1,β1] (∑_{β2} B[β1][α2,β2] x[σᵢ[β1,β2]]) 
                = ∑_{β1} A[α1,β1] Z[α2,β1] 
"""
function LinearAlgebra.mul!(y::AbstractVector, 
    C::WarpedTensorProductMap, x::AbstractVector)
    
    LinearMaps.check_dim_mul(y, C, x)
    @unpack A, B, σᵢ, σₒ = C
    (M1,M2) = size(σₒ)
    N1 = size(σᵢ,1)
    N2 = [count(a -> a>0, σᵢ[β1,:]) for β1 in 1:N1]

    Z = zeros(Float64, M2, N1)
    for α2 in 1:M2, β1 in 1:N1
        @simd for β2 in 1:N2[β1]
            @muladd Z[α2,β1] = Z[α2,β1] + B[β1][α2,β2]*x[σᵢ[β1,β2]]
        end
    end

    for α1 in 1:M1, α2 in 1:M2
        y[σₒ[α1,α2]] = 0.0
        @simd for β1 in 1:N1
            @muladd y[σₒ[α1,α2]] =  y[σₒ[α1,α2]] + A[α1,β1]*Z[α2,β1]
        end
    end

    return y
end

function LinearMaps._unsafe_mul!(y::AbstractVector, 
    transC::LinearMaps.TransposeMap{Float64, <:WarpedTensorProductMap},
    x::AbstractVector)

    LinearMaps.check_dim_mul(y, transC, x)
    @unpack A, B, σᵢ, σₒ = transC.lmap

    (N1,N2) = size(σₒ)
    M1 = size(σᵢ,1)
    M2 = [count(a -> a>0, σᵢ[α1,:]) for α1 in 1:M1]

    Z = zeros(Float64, M1, N2)
    for α1 in 1:M1, β2 in 1:N2
        @simd for β1 in 1:N1
            @muladd Z[α1,β2] = Z[α1,β2] + A[β1,α1]*x[σₒ[β1,β2]]
        end
    end

    for α1 in 1:M1
        for α2 in 1:M2[α1]
            y[σᵢ[α1,α2]] = 0.0
            @simd for β2 in 1:N2
                @muladd y[σᵢ[α1,α2]] = y[σᵢ[α1,α2]] + B[α1][β2,α2]*Z[α1,β2]
            end
        end
    end

    return y
end