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

    @unpack VOL, FAC, SRC, M, V = operators

    N_q = size(f[1],1)
    N_c = size(f[1],2)
    N_p = size(M,1)

    rhs_q = zeros(N_q, N_c)
    rhs = Matrix{Float64}(undef, N_p, N_c)

    @inbounds for e in 1:N_c
        @timeit thread_timer() "volume terms" @inbounds for m in 1:d
            rhs_q[:,e] = rhs_q[:,e] + VOL[m] * f[m][:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs_q[:,e] = rhs_q[:,e] + FAC * f_fac[:,e]
        end

        @timeit thread_timer() "source terms" begin
            rhs_q[:,e] = rhs_q[:,e] + SRC * s[:,e]
        end

        @timeit thread_timer() "mul test function" begin
            rhs[:,e] = V' * rhs_q[:,e]
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

    @unpack VOL, FAC, M, V = operators
    N_q = size(f[1],1)
    N_c = size(f[1],2)
    N_p = size(M,1)

    rhs_q = zeros(N_q, N_c)
    rhs = Matrix{Float64}(undef, N_p, N_c)

    @inbounds for e in 1:N_c
        @timeit thread_timer() "volume terms" @inbounds for m in 1:d
            rhs_q[:,e] = rhs_q[:,e] + VOL[m] * f[m][:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs_q[:,e] = rhs_q[:,e] + FAC * f_fac[:,e]
        end

        @timeit thread_timer() "mul test function" begin
            rhs[:,e] = V' * rhs_q[:,e]
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
    N_q = size(f[1],1)
    N_c = size(f[1],2)
    N_p = size(M,1)

    rhs_q = zeros(N_q, N_c)
    rhs = Matrix{Float64}(undef, N_p, N_c)

    @inbounds for e in 1:N_c
        @timeit thread_timer() "volume terms" begin
            rhs_q[:,e] = rhs_q[:,e] - VOL[m] * u[:,e]
        end

        @timeit thread_timer() "facet terms" begin
            rhs_q[:,e] = rhs_q[:,e] - FAC * u_fac[:,e]
        end

        @timeit thread_timer() "mul test function" begin
            rhs[:,e] = V' * rhs_q[:,e]
        end
    end

    @timeit thread_timer() "mass matrix solve" return M \ rhs
end