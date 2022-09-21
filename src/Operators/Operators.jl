module Operators

    using LinearAlgebra, LinearMaps
    using UnPack

    export TensorProductMap, SelectionMap, flux_diff, PartialTensorProductMap
    
    const Operator1D{T} = Union{UniformScaling{Bool}, 
        Matrix{T}, Transpose{T,Matrix{T}}}

    const NotIdentity{T} = Union{Matrix{T}, Transpose{T,Matrix{T}}}

    struct TensorProductMap{T} <: LinearMaps.LinearMap{T}
        A::Operator1D{T}
        B::Operator1D{T}
        σᵢ::Matrix{Int}
        σₒ::Matrix{Int}
    end

    Base.size(C::TensorProductMap) = (size(C.σₒ,1)*size(C.σₒ,2), 
        size(C.σᵢ,1)*size(C.σᵢ,2))

    function LinearAlgebra.mul!(y::AbstractVector, 
        C::TensorProductMap, x::AbstractVector)
        
        LinearMaps.check_dim_mul(y, C, x)
        @unpack A, B, σᵢ, σₒ = C

        return tensor_mul!(y,A,B,σᵢ,σₒ,x)
    end

    """
    Compute transpose using
    (A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ 
    """
    function LinearAlgebra.transpose(C::TensorProductMap)
        return TensorProductMap(transpose(C.A), transpose(C.B), C.σₒ, C.σᵢ)
    end

    """
    Multiply a vector of length N1*N2 by the M1*M2 x N1*N2 matrix

    ``C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] B[α2,β2]``

    or C = A ⊗ B. The action of this matrix on a vector ``x`` is 

    ``(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] B[α2,β2] x[σᵢ[β1,β2]] ``
    ``                = ∑_{β1} A[α1,β1] (∑_{β2} B[α2,β2] x[σᵢ[β1,β2]]) ``
    ``                = ∑_{β1} A[α1,β1] Z[α2,β1] ``
    """
    function tensor_mul!(y::AbstractVector, 
        A::NotIdentity{T}, B::NotIdentity{T},
        σᵢ::Matrix{Int}, σₒ::Matrix{Int},
        x::AbstractVector) where {T}

        (M1,M2) = size(σₒ)
        (N1,N2) = size(σᵢ)

        Z = Matrix{T}(zeros,M2,N1)
        @inbounds for α2 in 1:M2, β1 in 1:N1
            @inbounds @simd for β2 in 1:N2
                Z[α2,β1] += B[α2,β2]*x[σᵢ[β1,β2]]
            end
        end

        @inbounds for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = 0.0
            @inbounds @simd for β1 in 1:N1
                y[σₒ[α1,α2]] += A[α1,β1]*Z[α2,β1]
            end
        end

        return y
    end

    """
    Multiply a vector of length N1*N2 by the M1*M2 x N1*M2 matrix

    ``C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] δ_{α2,β2}``

    or C = A ⊗ I_{M2}. The action of this matrix on a vector ``x`` is 

    ``(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] δ_{α2,β2} x[σᵢ[β1,β2]] ``
    ``                = ∑_{β1} A[α1,β1] x[σᵢ[β1,α2]] ``
    """
    function tensor_mul!(y::AbstractVector, 
        A::NotIdentity, ::UniformScaling,
        σᵢ::Matrix{Int}, σₒ::Matrix{Int},
        x::AbstractVector)

        (M1,M2) = size(σₒ)
        (N1,N2) = size(σᵢ)

        @inbounds for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = 0.0
            @inbounds @simd for β1 in 1:N1
                y[σₒ[α1,α2]] += A[α1,β1]*x[σᵢ[β1,α2]]
            end
        end
        return y
    end

    """
    Multiply a vector of length N1*N2 by the M1*M2 x M1*N2 matrix
    ``C[σₒ[α1,α2], σᵢ[β1,β2]] = δ_{α1,β1} B[α2,β2]``

    or C = I_{M1} ⊗ B. The action of this matrix on a vector ``x`` is 

    ``(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} δ_{α1,β1} B[α2,β2] x[σᵢ[β1,β2]] ``
    ``                = ∑_{β2} B[α2,β2] x[σᵢ[α1,β2]]) ``
    """
    function tensor_mul!(y::AbstractVector, 
        ::UniformScaling, B::NotIdentity,
        σᵢ::Matrix{Int}, σₒ::Matrix{Int},
        x::AbstractVector)

        (M1,M2) = size(σₒ)
        (N1,N2) = size(σᵢ)

        @inbounds for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = 0.0
            @inbounds @simd for β2 in 1:N2
                y[σₒ[α1,α2]] += B[α2,β2]*x[σᵢ[α1,β2]]
            end
        end
        return y
    end

    """
    Select the nodes corresponding to `facet_node_ids` from a vector of volume nodes (avoids multiplication by zeros)
    """
    struct SelectionMap{T} <: LinearMaps.LinearMap{T}
        facet_ids::Vector{Int}
        volume_ids::Tuple{Vararg{Vector{Int}}}
        N_vol::Int
    end

    function SelectionMap(facet_ids::Vector{Int}, N_vol::Int)
        return SelectionMap{Float64}(facet_ids, 
            Tuple(findall(j->j==i, facet_ids) for i in 1:N_vol),
            N_vol)
    end

    Base.size(R::SelectionMap) = (length(R.facet_ids),R.N_vol)

    function LinearAlgebra.mul!(y::AbstractVector, 
        R::SelectionMap, x::AbstractVector)
        
        LinearMaps.check_dim_mul(y, R, x)
        @unpack facet_ids = R
        y[:] = x[facet_ids]
        return y
    end

    function LinearMaps._unsafe_mul!(y::AbstractVector, 
        transR::LinearMaps.TransposeMap{T, SelectionMap{T}},
        x::AbstractVector) where {T}
        
        LinearMaps.check_dim_mul(y, transR, x)
        @unpack volume_ids, facet_ids, N_vol = transR.lmap
        @inbounds for i in 1:N_vol
            if isempty(volume_ids[i])
                y[i] = 0.0
            else
                y[i] = sum(x[j] for j in volume_ids[i])
            end
        end
        return y
    end

    """
    Compute the flux-differencing term (D ⊙ F)1 
    """
    function flux_diff(D::LinearMaps.WrappedMap, F::AbstractArray{Float64,3})
        
        N_p = size(D,1)
        N_eq = size(F,3)
        y = Matrix{Float64}(undef,N_p,N_eq)
        
        @inbounds for l in 1:N_eq, i in 1:N_p
            y[i,l] = dot(D.lmap[i,:], F[i,:,l])
        end
        return 2.0*y
    end


    """
    Partial/"warped" tensor-product mapping

    The primal operator takes a partially tenorized vector to a fully tensorized vector

    σᵢ[]
    
    """    
    struct PartialTensorProductMap{T} <: LinearMaps.LinearMap{T}
        A::Operator1D{T}
        B::Vector{<:Operator1D{T}}
        σᵢ::Matrix{Int}
        σₒ::Matrix{Int}
    end


    Base.size(C::PartialTensorProductMap) = (count(a->a>0,C.σₒ), 
        count(a->a>0,C.σᵢ))

    function LinearAlgebra.mul!(y::AbstractVector, 
        C::PartialTensorProductMap, x::AbstractVector)
        
        LinearMaps.check_dim_mul(y, C, x)
        @unpack A, B, σᵢ, σₒ = C

        return tensor_mul!(y,A,B,σᵢ,σₒ,x)
    end

    """
    Multiply a vector of length Σ_{β1}N2[β1] by the M1*M2 x Σ_{β1}N2[β1] matrix

    ``C[σₒ[α1,α2], σᵢ[β1,β2]] = A[α1,β1] B[β1][α2,β2].``

    The action of this matrix on a vector ``x`` is 

    ``(Cx)[σₒ[α1,α2]] = ∑_{β1,β2} A[α1,β1] B[β1][α2,β2] x[σᵢ[β1,β2]] ``
    ``                = ∑_{β1} A[α1,β1] (∑_{β2} B[β1][α2,β2] x[σᵢ[β1,β2]]) ``
    ``                = ∑_{β1} A[α1,β1] Z[α2,β1] ``
    """
    function tensor_mul!(y::AbstractVector, 
        A::NotIdentity{T}, B::Vector{<:NotIdentity{T}},
        σᵢ::Matrix{Int}, σₒ::Matrix{Int},
        x::AbstractVector) where {T}

        (M1,M2) = size(σₒ)
        N1 = size(σᵢ,1)
        N2 = [count(a -> a>0, σᵢ[β1,:]) for β1 in 1:N1]

        Z = zeros(Float64, M2, N1)
        @inbounds for α2 in 1:M2, β1 in 1:N1
            @inbounds @simd for β2 in 1:N2[β1]
                Z[α2,β1] += B[β1][α2,β2]*x[σᵢ[β1,β2]]
            end
        end

        @inbounds for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = 0.0
            @inbounds @simd for β1 in 1:N1
                y[σₒ[α1,α2]] += A[α1,β1]*Z[α2,β1]
            end
        end

        return y
    end
    
end