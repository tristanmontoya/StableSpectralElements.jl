# Warped tensor-product operator (e.g. for Dubiner-type bases)
struct WarpedTensorProductMap3D{A_type, B_type, C_type, σᵢ_type, σₒ_type} <:
       LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    C::C_type
    σᵢ::σᵢ_type
    σₒ::σₒ_type
    N2::Vector{Int}
    N3::Matrix{Int}
    size::NTuple{2, Int}

    function WarpedTensorProductMap3D(A::A_type,
            B::B_type,
            C::C_type,
            σᵢ::σᵢ_type,
            σₒ::σₒ_type) where {A_type, B_type, C_type, σᵢ_type,
            σₒ_type}
        return new{A_type, B_type, C_type, σᵢ_type, σₒ_type}(A,
            B,
            C,
            σᵢ,
            σₒ,
            [count(a -> a > 0,
                 σᵢ[β1, 1, :])
             for β1 in axes(σᵢ, 1)],
            [count(a -> a > 0,
                 σᵢ[β1, β2, :])
             for β1 in axes(σᵢ, 1),
            β2 in axes(σᵢ, 2)],
            (size(A, 1) * size(B, 1) *
             size(C, 1),
                count(a -> a > 0, σᵢ)))
    end
end

@inline Base.size(L::WarpedTensorProductMap3D) = L.size

# Evaluate the matrix-vector product
# (Lx)[σₒ[α1,α2,α3]] = ∑_{β1,β2,β3} A[α1,β1] B[α2,β1,β2] C[α3,β1,β2,β3] 
#     * x[σᵢ[β1,β2,β3]]
#                    = ∑_{β1,β2} A[α1,β1] B[α2,β1,β2]
#     * (∑_{β3} C[α3,β1,β2,β3] x[σᵢ[β1,β2,β3]])
#                    = ∑_{β1,β2} A[α1,β1] B[α2,β1,β2] Z[β1,β2,α3]
#                    = ∑_{β1} A[α1,β1] (∑_{β2} B[α2,β1,β2] Z[β1,β2,α3])
#                    = ∑_{β1} A[α1,β1] W[β1,α2,α3]
@inline function LinearAlgebra.mul!(y::AbstractVector,
        L::WarpedTensorProductMap3D,
        x::AbstractVector)
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ, N2, N3) = L

    # these will be stack allocated
    Z = MArray{Tuple{size(σᵢ, 1), size(σᵢ, 2), size(σₒ, 3)}, Float64}(undef)
    W = MArray{Tuple{size(σᵢ, 1), size(σₒ, 2), size(σₒ, 3)}, Float64}(undef)

    @inbounds for β1 in axes(σᵢ, 1)
        for β2 in 1:N2[β1], α3 in axes(σₒ, 3)
            temp = 0.0
            @simd for β3 in 1:N3[β1, β2]
                @muladd temp = temp + C[α3, β1, β2, β3] * x[σᵢ[β1, β2, β3]]
            end
            Z[β1, β2, α3] = temp
        end
    end

    @inbounds for β1 in axes(σᵢ, 1), α2 in axes(σₒ, 2), α3 in axes(σₒ, 3)
        temp = 0.0
        @simd for β2 in 1:N2[β1]
            @muladd temp = temp + B[α2, β1, β2] * Z[β1, β2, α3]
        end
        W[β1, α2, α3] = temp
    end

    @inbounds for α1 in axes(σₒ, 1), α2 in axes(σₒ, 2), α3 in axes(σₒ, 3)
        temp = 0.0
        @simd for β1 in axes(σᵢ, 1)
            @muladd temp = temp + A[α1, β1] * W[β1, α2, α3]
        end
        y[σₒ[α1, α2, α3]] = temp
    end

    return y
end

# Evaluate the matrix-vector product
# (Lx)[σᵢ[β1,β2,β3]] = ∑_{α1,α2,α3} C[α3,β1,β2,β3] B[α2,β1,β2] A[α1,β1]
#    * x[σₒ[α1,α2,α3]]
#                    = ∑_{α2,α3} C[α3,β1,β2,β3] B[α2,β1,β2]
#    * (∑_{α1} A[α1,β1] x[σₒ[α1,α2,α3]])
#                    = ∑_{α2,α3} C[α3,β1,β2,β3] B[α2,β1,β2] W[β1,α2,α3]
#                    = ∑_{α3} C[α3,β1,β2,β3] (∑_{α2} B[α2,β1,β2] W[β1,α2,α3])
#                    = ∑_{α3} C[α3,β1,β2,β3] Z[β1,β2,α3]
@inline function LinearMaps._unsafe_mul!(y::AbstractVector,
        L::LinearMaps.TransposeMap{Float64,
            <:WarpedTensorProductMap3D},
        x::AbstractVector)
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ, N2, N3) = L.lmap

    # these will be stack allocated
    Z = MArray{Tuple{size(σᵢ, 1), size(σᵢ, 2), size(σₒ, 3)}, Float64}(undef)
    W = MArray{Tuple{size(σᵢ, 1), size(σₒ, 2), size(σₒ, 3)}, Float64}(undef)

    @inbounds for β1 in axes(σᵢ, 1), α2 in axes(σₒ, 2), α3 in axes(σₒ, 3)
        temp = 0.0
        @simd for α1 in axes(σₒ, 1)
            @muladd temp = temp + A[α1, β1] * x[σₒ[α1, α2, α3]]
        end
        W[β1, α2, α3] = temp
    end

    @inbounds for β1 in axes(σᵢ, 1)
        for β2 in 1:N2[β1], α3 in axes(σₒ, 3)
            temp = 0.0
            @simd for α2 in axes(σₒ, 2)
                @muladd temp = temp + B[α2, β1, β2] * W[β1, α2, α3]
            end
            Z[β1, β2, α3] = temp
        end
    end

    @inbounds for β1 in axes(σᵢ, 1)
        for β2 in 1:N2[β1]
            for β3 in 1:N3[β1, β2]
                temp = 0.0
                @simd for α3 in axes(σₒ, 3)
                    @muladd temp = temp + C[α3, β1, β2, β3] * Z[β1, β2, α3]
                end
                y[σᵢ[β1, β2, β3]] = temp
            end
        end
    end

    return y
end
