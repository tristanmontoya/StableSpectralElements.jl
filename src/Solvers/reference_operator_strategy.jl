function Solver(conservation_law::AbstractConservationLaw,     
    spatial_discretization::SpatialDiscretization,
    form::AbstractResidualForm,
    strategy::ReferenceOperator)

    operators = make_operators(spatial_discretization, form)
    return Solver(conservation_law, 
        operators,
        spatial_discretization.mesh.xyzq,
        spatial_discretization.mesh.mapP, form, strategy)
end

function apply_operators(
    operators::DiscretizationOperators{d},
    f::NTuple{d,Matrix{Float64}}, 
    f_fac::Matrix{Float64}, 
    ::ReferenceOperator,
    s::Matrix{Float64}) where {d}

    @unpack VOL, FAC, SRC, M = operators
    N_eq = size(f[1],2)
    rhs = zeros(size(VOL[1],1), N_eq)

    @inbounds for e in 1:N_eq
        @timeit thread_timer() "volume terms" @inbounds for m in 1:d
            rhs[:,e] = rhs[:,e] + VOL[m] * f[m][:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs[:,e] = rhs[:,e] + FAC * f_fac[:,e]
        end

        @timeit thread_timer() "source terms" begin
            rhs[:,e] = rhs[:,e] + SRC * s[:,e]
        end
    end

    @timeit thread_timer() "mass matrix solve" return M \ rhs
end

function apply_operators(
    operators::DiscretizationOperators{d},
    f::NTuple{d,Matrix{Float64}}, 
    f_fac::Matrix{Float64}, 
    ::ReferenceOperator,
    s::Nothing) where {d}

    @unpack VOL, FAC, SRC, M = operators
    N_eq = size(f[1],2)
    rhs = zeros(size(VOL[1],1), N_eq)

    @inbounds for e in 1:N_eq
        @timeit thread_timer() "volume terms" @inbounds for m in 1:d
            rhs[:,e] = rhs[:,e] + VOL[m] * f[m][:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs[:,e] = rhs[:,e] + FAC * f_fac[:,e]
        end
    end

    @timeit thread_timer() "mass matrix solve" return M \ rhs
end

function auxiliary_variable(m::Int, 
    operators::DiscretizationOperators{d},
    u::Matrix{Float64},
    u_fac::Matrix{Float64}, 
    ::ReferenceOperator) where {d}

    @unpack VOL, FAC, M = operators
    N_eq = size(u,2)
    rhs = zeros(size(VOL[1],1), N_eq)

    @inbounds for e in 1:N_eq
        @timeit thread_timer() "volume terms" begin
            rhs[:,e] = rhs[:,e] - VOL[m] * u[:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs[:,e] = rhs[:,e] - FAC * u_fac[:,e]
        end
    end

    @timeit thread_timer() "mass matrix solve" return M \ rhs
end