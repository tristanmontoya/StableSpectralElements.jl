import Base: transpose, *

struct SplitSimplexMap{D_type, Dt_type, Hinv_type, E_type, L_type, map_type} <: LinearMaps.LinearMap{Float64}
    D   :: D_type
    Dt  :: Dt_type
    Hinv   :: Hinv_type
    E   :: E_type
    L   :: L_type
    t1  :: Vector{Float64}
    t2 :: Vector{Float64}
    t3 :: Vector{Float64}
    T :: Matrix{Float64}
    map :: map_type
    m   :: Int
    dim :: Int
    nloc:: Int 
    size :: NTuple{2, Int}

    # inner constructor
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
        T = zeros(Float64, nloc, nloc) # temp matrix for tensor product mat-vec
        return new{D_type, Dt_type, Hinv_type, E_type, L_type, map_type}(D, Dt, Hinv, E, L, t1, t2, t3, T, map, m, dim, nloc, size)   # type parameters inferred automatically
    end
end

# wrapper struct for the transpose 
# field parent points to the original SplitSimplexMap
# Same thing as if we did transpose(A) where Q is a Matrix{Float64}, so tranpose(A) is a type Transpose{Float64, Matrix{Float64})}
# we need this to define the mul! method for the transpose, ie be able to select which mul!() method to call
struct SplitSimplexMapT
    parent::SplitSimplexMap
end

# this function says that when we call tranpose(A) where A is a SplitSimplexMap, we get back a SplitSimplexMapT struct
# ie wrap the input in the SplitSimplexMap transpose struct
@inline transpose(M::SplitSimplexMap) = SplitSimplexMapT(M)

@inline Base.size(L::SplitSimplexMap) = L.size # define size method

# define mul! method
using StaticArrays, LinearAlgebra

@inline function LinearAlgebra.mul!(y::AbstractVector, M::SplitSimplexMap, x::AbstractVector)
    (; D, Dt, Hinv, E, L, t1, t2, t3, map, m, dim) = M
    fill!(y, 0.0)

    @inbounds for k = 1:dim+1
        idx = map[k]
        F = @view x[idx]

        fill!(t3, 0.0)  # zero accumulator for this subdomain

        @simd for j = 1:dim
            Lkjm = @view L[:,(k-1)*dim^2 + (j-1)*dim + m]
            LinearMaps.mul!(t1,D[j],F) 
            t1 .*= Lkjm
            t2 .= Lkjm .* F 
            LinearMaps.mul!(t2,Dt[j],t2) 
            t3 .+= 0.5 .* (t1 .- t2)
        end

        # surface term reuse t1
        t1 .= E[:, (k-1)*dim+m] .* F 
        t3 .+= 0.5 .* t1 # add contribution from the surface integral (dot is for elementwise addition) 
        
        y[idx] .+= t3 # left multiply by P
    end

    @. y *= Hinv

    return y
end

# for the transpose
@inline function LinearAlgebra.mul!(y::AbstractArray, Mᵀ::SplitSimplexMapT, x::AbstractArray)
    M = Mᵀ.parent
    (; D, Dt, Hinv, E, L, t1, t2, t3, map, m, dim) = M

    fill!(y, 0.0)

    @inbounds for k = 1:dim+1
        idx = map[k]

        F = Hinv .*x
        F = @view F[idx]
        
        fill!(t3, 0.0)

        @simd for j = 1:dim
            Lkjm = @view L[:,(k-1)*dim^2 + (j-1)*dim + m]
            t1 .= Lkjm .* F
            LinearMaps.mul!(t1, Dt[j], t1)
            LinearMaps.mul!(t2, D[j], F)
            t2 .= Lkjm .* t2
            t3 .+= 0.5 .* (t1 .- t2)
        end

        t1 .= E[:,(k-1)*dim+m] .* F
        t3 .+= 0.5 .* t1

        y[idx] .+= t3
    end

    return y
end