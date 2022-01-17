module Operators

    using LinearAlgebra, LinearMaps
    using UnPack

    export TensorProductMap, SelectionMap
    
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
        C::TensorProductMap,
        x::AbstractVector)
        
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

        Z = Matrix{T}(undef,M2,N1)
        for α2 in 1:M2, β1 in 1:N1
            Z[α2,β1] = sum(B[α2,β2]*x[σᵢ[β1,β2]] for β2 in 1:N2)
        end

        for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = sum(A[α1,β1]*Z[α2,β1] for β1 in 1:N1)
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

        for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = sum(A[α1,β1]*x[σᵢ[β1,α2]] for β1 in 1:N1)
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

        for α1 in 1:M1, α2 in 1:M2
            y[σₒ[α1,α2]] = sum(B[α2,β2]*x[σᵢ[α1,β2]] for β2 in 1:N2)
        end
        return y
    end

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
        R::SelectionMap,
        x::AbstractVector)
        
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
        for i in 1:N_vol
            if isempty(volume_ids[i])
                y[i] = 0.0
            else
                y[i] = sum(x[j] for j in volume_ids[i])
            end
        end
        return y
    end
#=
    function LinearMaps._unsafe_mul!(y::AbstractMatrix, 
        transR::LinearMaps.TransposeMap{T, SelectionMap{T}},
        x::AbstractVector) where {T}
        
        LinearMaps.check_dim_mul(y, transR, x)
        @unpack volume_ids, facet_ids, N_vol = transR.lmap
        for k in 1:size(y,2)
            for i in 1:N_vol
                if isempty(volume_ids[i])
                    y[i,k] = 0.0
                else
                    y[i,k] = sum(x[j,k] for j in volume_ids[i])
                end
            end
        end
        return y
    end
=#
end
