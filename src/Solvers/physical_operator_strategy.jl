function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    strategy::PhysicalOperator)

    operators = make_operators(spatial_discretization, form)

    return Solver(conservation_law, 
            [precompute(operators[k]) 
                for k in 1:spatial_discretization.N_e],
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

function apply_operators(
    operators::DiscretizationOperators{d},
    f::NTuple{d,Matrix{Float64}}, 
    f_fac::Matrix{Float64}, 
    ::PhysicalOperator,
    s::Union{Matrix{Float64},Nothing}) where {d}

    
    @unpack VOL, FAC, SRC, M = operators
    N_c = size(f[1],2)
    rhs = zeros(size(VOL[1],1), N_c)

    @inbounds for e in 1:N_c
        @timeit thread_timer() "volume terms" @inbounds for m in 1:d
            rhs[:,e] = rhs[:,e] + VOL[m] * f[m][:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs[:,e] = rhs[:,e] + FAC * f_fac[:,e]
        end

        if !isnothing(s)
            @timeit thread_timer() "source terms" begin
                rhs[:,e] = rhs[:,e] + SRC * s[:,e]
            end
        end
    end

    return rhs
end

function auxiliary_variable(m::Int, 
    operators::DiscretizationOperators{d},
    u::Matrix{Float64},
    u_fac::Matrix{Float64}, 
    ::PhysicalOperator) where {d}

    @unpack VOL, FAC, M = operators
    N_c = size(u,2)
    rhs = zeros(size(VOL[1],1), N_c)

    @inbounds for e in 1:N_c
        @timeit thread_timer() "volume terms" begin
            rhs[:,e] = rhs[:,e] - VOL[m] * u[:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs[:,e] = rhs[:,e] - FAC * u_fac[:,e]
        end
    end

    return rhs
end