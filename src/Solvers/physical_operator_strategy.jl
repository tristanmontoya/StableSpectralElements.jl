function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    strategy::PhysicalOperator)

    operators = make_operators(spatial_discretization, form)

    return Solver(conservation_law, 
            [precompute(operators[k]) 
                for k in 1:spatial_discretization.N_el],
            spatial_discretization.mesh.xyzq,
            spatial_discretization.mesh.mapP, form, strategy)
end

function precompute(operators::DiscretizationOperators{d}) where {d}
    @unpack VOL, FAC, SRC, M, V, Vf, scaled_normal = operators

    #precompute the physical mass matrix inverse
    inv_M = inv(M)

    return DiscretizationOperators{d}(
        Tuple(combine(inv_M*VOL[n]) for n in 1:d),
        combine(inv_M*FAC), 
        combine(inv_M*SRC),
        M, V, Vf, scaled_normal)
end

function apply_operators!(residual::Matrix{Float64},
    operators::DiscretizationOperators{d},  
    f::NTuple{d,Matrix{Float64}}, 
    f_fac::Matrix{Float64}, ::PhysicalOperator,
    s::Union{Matrix{Float64},Nothing}) where {d}

    @unpack VOL, FAC, SRC = operators
    
    rhs = zero(residual) # only allocation

    @timeit thread_timer() "volume terms" begin
        @inbounds for m in 1:d
            rhs += mul!(residual, VOL[m], f[m])
        end
    end

    @timeit thread_timer() "facet terms" begin
        rhs += mul!(residual, FAC, f_fac)
    end

    if !isnothing(s)
        @timeit thread_timer() "source terms" begin
            rhs += mul!(residual, SRC, s)
        end
    end

    # no mass matrix solve needed
    residual = rhs

    return residual
end

function auxiliary_variable!(m::Int, 
    q::Matrix{Float64},
    operators::DiscretizationOperators{d},
    u::Matrix{Float64},
    u_fac::Matrix{Float64}, 
    ::PhysicalOperator) where {d}

    @unpack VOL, FAC = operators
    
    rhs = similar(q) # only allocation

    @timeit thread_timer() "volume terms" begin
        mul!(rhs, -VOL[m], u)
    end

    @timeit thread_timer() "facet terms" begin
        rhs += mul!(q, -FAC, u_fac)
    end

    # no mass matrix solve needed
    q = rhs

    return q
end