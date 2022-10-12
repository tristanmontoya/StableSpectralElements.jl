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

    Z = Matrix{Float64}(undef, size(σₒ,2), size(σᵢ,1))
    @inbounds for α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        N2 = count(a -> a>0, σᵢ[β1,:])
        temp = 0.0
        @inbounds @simd for β2 in 1:N2
            @muladd temp = temp + B[β1][α2,β2] * x[σᵢ[β1,β2]]
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

function LinearMaps._unsafe_mul!(y::AbstractVector, 
    transC::LinearMaps.TransposeMap{Float64, <:WarpedTensorProductMap},
    x::AbstractVector)

    LinearMaps.check_dim_mul(y, transC, x)
    @unpack A, B, σᵢ, σₒ = transC.lmap

    Z = Matrix{Float64}(undef, size(σᵢ,1), size(σₒ,2))
    @inbounds for α1 in axes(σᵢ,1), β2 in axes(σₒ,2)
        temp = 0.0
        @inbounds @simd for β1 in axes(σₒ,1)
            @muladd temp = temp + A[β1,α1]*x[σₒ[β1,β2]]
        end
        Z[α1,β2] = temp
    end

    @inbounds for α1 in axes(σᵢ,1)
        M2 = count(a -> a>0, σᵢ[α1,:])
        @inbounds for α2 in 1:M2
            temp = 0.0
            @inbounds @simd for β2 in axes(σₒ,2)
                @muladd temp = temp + B[α1][β2,α2]*Z[α1,β2]
            end
            y[σᵢ[α1,α2]] = temp
        end
    end

    return y
end