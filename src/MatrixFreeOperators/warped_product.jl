"""
Warped tensor-product operator (e.g. for Dubiner-type bases)
"""    
struct WarpedTensorProductMap{A_type, B_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::Vector{B_type}
    σᵢ::Matrix{Int}
    σₒ::Matrix{Int}
end

@inline Base.size(C::WarpedTensorProductMap) = (count(a->a>0,C.σₒ), 
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

    Z = Matrix{Float64}(undef, M2, N1)
    @inbounds for α2 in 1:M2, β1 in 1:N1
        Zij = 0.0
        @turbo for β2 in 1:N2[β1]
            @muladd Zij = Zij + B[β1][α2,β2]*x[σᵢ[β1,β2]]
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

function LinearMaps._unsafe_mul!(y::AbstractVector, 
    transC::LinearMaps.TransposeMap{Float64, <:WarpedTensorProductMap},
    x::AbstractVector)

    LinearMaps.check_dim_mul(y, transC, x)
    @unpack A, B, σᵢ, σₒ = transC.lmap

    (N1,N2) = size(σₒ)
    M1 = size(σᵢ,1)
    M2 = [count(a -> a>0, σᵢ[α1,:]) for α1 in 1:M1]

    Z = Matrix{Float64}(undef, M1, N2)
    @turbo for α1 in 1:M1, β2 in 1:N2
        Zij = 0.0
        for β1 in 1:N1
            @muladd Zij = Zij + A[β1,α1]*x[σₒ[β1,β2]]
        end
        Z[α1,β2] = Zij
    end

    @inbounds for α1 in 1:M1
        @turbo for α2 in 1:M2[α1]
            yi = 0.0
            for β2 in 1:N2
                @muladd yi = yi + B[α1][β2,α2]*Z[α1,β2]
            end
            y[σᵢ[α1,α2]] = yi
        end
    end

    return y
end