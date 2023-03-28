struct WeightAdjustedMap <: LinearMaps.LinearMap{Float64}
    V::LinearMap
    W::Diagonal
    Jinv::Diagonal
    M::Union{UniformScaling,Diagonal,LinearMap}
    M_inv::Union{UniformScaling,Diagonal,Matrix}

    function WeightAdjustedMap(V, W, Jinv, tol=1.0e-13)
        M_mat = Matrix(V'*W*V)
        M_diag = diag(M_mat)
        if maximum(abs.(M_mat - diagm(M_diag))) < tol
            if maximum(abs.(M_diag .- 1.0)) < tol
                M = I
                M_inv = I
            else
                M = Diagonal(M_diag)
                M_inv = inv(M)
            end
        else
            M = V' * W * V
            M_inv = inv(M_mat)
        end
        return new(V,W,Jinv,M,M_inv)
    end
end

function LinearAlgebra.mul!(y::AbstractVector{Float64}, 
    L::WeightAdjustedMap,
    x::AbstractVector{Float64})
    @unpack V, W, Jinv, M = L
    mul!(y, M*inv(V' * W * Jinv * V)*M,  x)
end

function LinearAlgebra.ldiv!(L::WeightAdjustedMap,
    rhs::AbstractArray{Float64})
    @unpack V, W, Jinv, M_inv = L
    y = similar(rhs)
    mul!(y, M_inv * V' * W * Jinv * V * M_inv, rhs)
    rhs[:] = y
end
