import Base: transpose

struct SplitSimplexMap{D_type, Dt_type, Hinv_type, E_type, L_type, map_type} <:
       LinearMaps.LinearMap{Float64}
    D::D_type
    Dt::Dt_type
    Hinv::Hinv_type
    E::E_type
    L::L_type
    t1::Vector{Float64}
    t2::Vector{Float64}
    t3::Vector{Float64}
    map::map_type
    m::Int
    dim::Int
    nloc::Int
    size::NTuple{2, Int}

    function SplitSimplexMap(
            D::D_type,
            Dt::Dt_type,
            Hinv::Hinv_type,
            E::E_type,
            L::L_type,
            map::map_type,
            m::Int,
            dim::Int,
            nloc::Int) where {D_type, Dt_type, Hinv_type, E_type, L_type, map_type}
        n = length(Hinv)
        size = (n, n)
        t1 = zeros(Float64, nloc)
        t2 = similar(t1)
        t3 = similar(t1)
        return new{D_type, Dt_type, Hinv_type, E_type, L_type, map_type}(
            D, Dt, Hinv, E, L, t1, t2, t3, map, m, dim, nloc, size)
    end
end

# Transpose wrapper for dispatching matrix-free multiplication
struct SplitSimplexMapT
    parent::SplitSimplexMap
end

@inline transpose(M::SplitSimplexMap) = SplitSimplexMapT(M)

@inline Base.size(L::SplitSimplexMap) = L.size

using LinearAlgebra

@inline function LinearAlgebra.mul!(
        y::AbstractVector, M::SplitSimplexMap, x::AbstractVector)
    (; D, Dt, Hinv, E, L, t1, t2, t3, map, m, dim) = M
    fill!(y, 0.0)

    @inbounds for k in 1:(dim + 1)
        idx = map[k]
        F = @view x[idx]

        fill!(t3, 0.0)

        @simd for j in 1:dim
            Lkjm = @view L[:, (k - 1) * dim^2 + (j - 1) * dim + m]
            LinearMaps.mul!(t1, D[j], F)
            t1 .*= Lkjm
            t2 .= Lkjm .* F
            LinearMaps.mul!(t2, Dt[j], t2)
            t3 .+= 0.5 .* (t1 .- t2)
        end

        # Reuse t1 for the surface term
        t1 .= E[:, (k - 1) * dim + m] .* F
        t3 .+= 0.5 .* t1

        y[idx] .+= t3
    end

    @. y *= Hinv

    return y
end

@inline function LinearAlgebra.mul!(
        y::AbstractArray, Mᵀ::SplitSimplexMapT, x::AbstractArray)
    M = Mᵀ.parent
    (; D, Dt, Hinv, E, L, t1, t2, t3, map, m, dim) = M

    fill!(y, 0.0)

    @inbounds for k in 1:(dim + 1)
        idx = map[k]

        F = Hinv .* x
        F = @view F[idx]

        fill!(t3, 0.0)

        @simd for j in 1:dim
            Lkjm = @view L[:, (k - 1) * dim^2 + (j - 1) * dim + m]
            t1 .= Lkjm .* F
            LinearMaps.mul!(t1, Dt[j], t1)
            LinearMaps.mul!(t2, D[j], F)
            t2 .= Lkjm .* t2
            t3 .+= 0.5 .* (t1 .- t2)
        end

        t1 .= E[:, (k - 1) * dim + m] .* F
        t3 .+= 0.5 .* t1

        y[idx] .+= t3
    end

    return y
end
