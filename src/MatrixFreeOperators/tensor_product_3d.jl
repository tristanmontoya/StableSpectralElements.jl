struct TensorProductMap3D{A_type,B_type,C_type} <: LinearMaps.LinearMap{Float64}
    A::A_type
    B::B_type
    C::C_type
    σᵢ::AbstractArray{Int,3}
    σₒ::AbstractArray{Int,3}
end

@inline Base.size(L::TensorProductMap3D) = (
    size(L.σₒ,1)*size(L.σₒ,2)*size(L.σₒ,3), 
    size(L.σᵢ,1)*size(L.σᵢ,2)*size(L.σᵢ,3))

function TensorProductMap3D(A, B, C)
    (M1,N1) = size(A)
    (M2,N2) = size(B)
    (M3,N3) = size(C)
    σₒ = SArray{Tuple{M1,M2,M3},Int}(
        [M2*M3*(α1-1) + M3*(α2-1) + α3 for α1 in 1:M1, α2 in 1:M2, α3 in 1:M3])
    σᵢ = SArray{Tuple{N1,N2,N3},Int}(
        [N2*N3*(β1-1) + N3*(β2-1) + β3 for β1 in 1:N1, β2 in 1:N2, β3 in 1:N3])

    if A isa LinearMaps.UniformScalingMap{Bool} 
        A = I 
    elseif A isa Union{LinearMaps.WrappedMap,OctavianMap}
        A = SMatrix{M1,N1,Float64}(A.lmap)
    end
    if B isa LinearMaps.UniformScalingMap{Bool} 
        B = I 
    elseif B isa Union{LinearMaps.WrappedMap,OctavianMap}
        B = SMatrix{M2,N2,Float64}(B.lmap)
    end
    if C isa LinearMaps.UniformScalingMap{Bool} 
        C = I 
    elseif C isa Union{LinearMaps.WrappedMap,OctavianMap}
        C = SMatrix{M3,N3,Float64}(C.lmap)
    end
    return TensorProductMap3D(A,B,C, σᵢ, σₒ)
end

"""
Compute transpose using
(A ⊗ B ⊗ C)ᵀ = Aᵀ ⊗ Bᵀ ⊗ Cᵀ
"""
function LinearAlgebra.transpose(L::TensorProductMap3D)
    return TensorProductMap3D(transpose(L.A), transpose(L.B), 
        transpose(L.C), L.σₒ, L.σᵢ)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:AbstractMatrix{Float64},<:AbstractMatrix{Float64},<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    Z_3 = MArray{Tuple{size(σᵢ,1), size(σᵢ,2), size(σₒ,3)},Float64}(undef)
    Z_2 = MArray{Tuple{size(σᵢ,1), size(σₒ,2), size(σₒ,3)},Float64}(undef)
    
    @inbounds for α3 in axes(σₒ,3), β2 in axes(σᵢ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β3 in axes(σᵢ,3)
            @muladd temp = temp + C[α3,β3] * x[σᵢ[β1,β2,β3]]
        end
        Z_3[β1,β2,α3] = temp
    end

    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * Z_3[β1,β2,α3]
        end
        Z_2[β1,α2,α3] = temp
    end

    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), α1 in axes(σₒ,1)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * Z_2[β1,α2,α3]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return y
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:AbstractMatrix{Float64},<:UniformScaling,<:UniformScaling},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L
    
    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2), α3 in axes(σₒ,3)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * x[σᵢ[β1,α2,α3]]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return lmul!(B*C,y)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:UniformScaling,<:AbstractMatrix{Float64},<:UniformScaling},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2), α3 in axes(σₒ,3)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[α1,β2,α3]]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return lmul!(A*C,y)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:UniformScaling,<:UniformScaling,<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})

    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    @inbounds for α1 in axes(σₒ,1), α2 in axes(σₒ,2), α3 in axes(σₒ,3)
        temp = 0.0
        @simd for β3 in axes(σᵢ,3)
            @muladd temp = temp + C[α3,β3] * x[σᵢ[α1,α2,β3]]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return lmul!(A*B,y)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:AbstractMatrix{Float64},<:AbstractMatrix{Float64},<:UniformScaling},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    Z_2 = MArray{Tuple{size(σᵢ,1), size(σₒ,2), size(σₒ,3)},Float64}(undef)
    
    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * x[σᵢ[β1,β2,α3]]
        end
        Z_2[β1,α2,α3] = temp
    end

    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), α1 in axes(σₒ,1)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * Z_2[β1,α2,α3]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return lmul!(C,y)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:AbstractMatrix{Float64},<:UniformScaling,<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    Z_3 = MArray{Tuple{size(σᵢ,1), size(σᵢ,2), size(σₒ,3)},Float64}(undef)
   
    @inbounds for α3 in axes(σₒ,3), β2 in axes(σᵢ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β3 in axes(σᵢ,3)
            @muladd temp = temp + C[α3,β3] * x[σᵢ[β1,β2,β3]]
        end
        Z_3[β1,β2,α3] = temp
    end

    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), α1 in axes(σₒ,1)
        temp = 0.0
        @simd for β1 in axes(σᵢ,1)
            @muladd temp = temp + A[α1,β1] * Z_3[β1,α2,α3]
        end
        y[σₒ[α1,α2,α3]] = temp
    end

    return lmul!(B, y)
end

@inline function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::TensorProductMap3D{<:UniformScaling,<:AbstractMatrix{Float64},<:AbstractMatrix{Float64}},
    x::AbstractVector{Float64})
    
    LinearMaps.check_dim_mul(y, L, x)
    (; A, B, C, σᵢ, σₒ) = L

    Z_3 = MArray{Tuple{size(σᵢ,1), size(σᵢ,2), size(σₒ,3)},Float64}(undef)
    
    @inbounds for α3 in axes(σₒ,3), β2 in axes(σᵢ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β3 in axes(σᵢ,3)
            @muladd temp = temp + C[α3,β3] * x[σᵢ[β1,β2,β3]]
        end
        Z_3[β1,β2,α3] = temp
    end

    @inbounds for α3 in axes(σₒ,3), α2 in axes(σₒ,2), β1 in axes(σᵢ,1)
        temp = 0.0
        @simd for β2 in axes(σᵢ,2)
            @muladd temp = temp + B[α2,β2] * Z_3[β1,β2,α3]
        end
        y[σₒ[β1,α2,α3]] = temp
    end

    return lmul!(A,y)
end