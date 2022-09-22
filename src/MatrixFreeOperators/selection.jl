"""
Select the nodes corresponding to `facet_node_ids` from a vector of volume nodes (avoids multiplication by zeros)
"""
struct SelectionMap{T} <: LinearMaps.LinearMap{T}
    facet_ids::Vector{Int} # vol node inds for each fac node
    volume_ids::Tuple{Vararg{Vector{Int}}} # fac node inds for each vol node
end

function SelectionMap(facet_ids::Vector{Int}, N_vol::Int)
    return SelectionMap{Float64}(facet_ids, 
        Tuple(findall(j->j==i, facet_ids) for i in 1:N_vol))
end

Base.size(R::SelectionMap) = (length(R.facet_ids),length(R.volume_ids))

function LinearAlgebra.mul!(y::AbstractVector, 
    R::SelectionMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, R, x)
    y[:] = x[R.facet_ids]
    return y
end

function LinearMaps._unsafe_mul!(y::AbstractVector, 
    transR::LinearMaps.TransposeMap{T, SelectionMap{T}},
    x::AbstractVector) where {T}
    
    LinearMaps.check_dim_mul(y, transR, x)
    N_vol = length(transR.lmap.volume_ids)
    
    @simd for i in 1:N_vol
        if isempty(transR.lmap.volume_ids[i])
            y[i] = 0.0
        else
            y[i] = sum(x[j] for j in transR.lmap.volume_ids[i])
        end
    end
    return y
end