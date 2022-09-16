module Solvers

    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, inv, mul!
    using TimerOutputs
    using LinearMaps: LinearMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve

    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, Hyperbolic, Parabolic, Mixed, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux, numerical_flux, LaxFriedrichsNumericalFlux, BR1

    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..ParametrizedFunctions: AbstractParametrizedFunction, AbstractParametrizedFunction, NoSourceTerm, evaluate
    using ..Operators: flux_diff

    export AbstractResidualForm, AbstractMappingForm, AbstractStrategy, PhysicaOperators, Eager, Lazy, Solver, StandardMapping, SkewSymmetricMapping, CreanMapping, initialize, semidiscretize, precompute, apply_operators!, auxiliary_variable!, combine, get_dof, rhs!

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end

    struct Eager <: AbstractStrategy end
    struct Lazy <: AbstractStrategy end

    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end

    struct PhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        SRC::LinearMap
        M::AbstractMatrix
        V::LinearMap
        Vf::LinearMap
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    struct Solver{d,ResidualForm,PDEType}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::Vector{PhysicalOperators}
        x_q::NTuple{d,Matrix{Float64}}
        connectivity::Matrix{Int}
        form::ResidualForm
        strategy::AbstractStrategy
    end

    function Solver(conservation_law::AbstractConservationLaw,     
        spatial_discretization::SpatialDiscretization,
        form::AbstractResidualForm,
        strategy::Lazy)
        return Solver(conservation_law, 
            make_operators(spatial_discretization, form),
            spatial_discretization.mesh.xyzq,
            spatial_discretization.mesh.mapP, form, strategy)
    end

    function Solver(conservation_law::AbstractConservationLaw,     
        spatial_discretization::SpatialDiscretization,
        form::AbstractResidualForm,
        strategy::Eager)

        operators = make_operators(spatial_discretization, form)
        return Solver(conservation_law, 
            [precompute(operators[k]) for k in 1:spatial_discretization.N_el],
                spatial_discretization.mesh.xyzq,
                spatial_discretization.mesh.mapP, form, strategy)
    end

    function initialize(initial_data::AbstractParametrizedFunction,
        conservation_law::AbstractConservationLaw,
        spatial_discretization::SpatialDiscretization{d}) where {d}

        @unpack M, geometric_factors = spatial_discretization
        @unpack N_q, V, W = spatial_discretization.reference_approximation
        @unpack xyzq = spatial_discretization.mesh
        N_p, N_eq, N_el = get_dof(spatial_discretization, conservation_law)
        
        u0 = Array{Float64}(undef, N_p, N_eq, N_el)
        for k in 1:N_el
            u0[:,:,k] = M[k] \ convert(Matrix, V' * W * 
                Diagonal(geometric_factors.J_q[:,k]) * 
                evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        end
        return u0
    end

    function semidiscretize(
        conservation_law::AbstractConservationLaw,spatial_discretization::SpatialDiscretization,
        initial_data::AbstractParametrizedFunction, 
        form::AbstractResidualForm,
        tspan::NTuple{2,Float64}, 
        strategy::AbstractStrategy)

        u0 = initialize(
            initial_data,
            conservation_law,
            spatial_discretization)

        return semidiscretize(Solver(conservation_law,spatial_discretization,
            form,strategy),u0, tspan)
    end

    function semidiscretize(solver::Solver, u0::Array{Float64,3},
        tspan::NTuple{2,Float64})
        return ODEProblem(rhs!, u0, tspan, solver)
    end

    function precompute(operators::PhysicalOperators{d}) where {d}
        @unpack VOL, FAC, SRC, M, V, Vf, NTR, scaled_normal = operators
        inv_M = inv(M)
        return PhysicalOperators(
            Tuple(combine(inv_M*VOL[n]) for n in 1:d),
            combine(inv_M*FAC), 
            combine(inv_M*SRC),
            M, V, Vf, scaled_normal)
    end

    function combine(operator::LinearMap)
        return LinearMap(convert(Matrix,operator))
    end

    function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        conservation_law::AbstractConservationLaw{d}) where {d}
        return (spatial_discretization.reference_approximation.N_p, 
            conservation_law.N_eq, spatial_discretization.N_el)
    end

    """
        Physical-operator form
    """
    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},  
        f::NTuple{d,Matrix{Float64}}, 
        f_fac::Matrix{Float64}, ::Eager,
        s::Union{Matrix{Float64},Nothing}) where {d}
        to = get_timer(string("thread_timer_", Threads.threadid()))
        
        @timeit to "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            @inbounds for m in 1:d
                volume_terms += mul!(residual, operators.VOL[m], f[m])
            end
        end

        @timeit to "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit to "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        return rhs
    end

    """
        Reference-operator form
    """
    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},
        f::NTuple{d,Matrix{Float64}}, 
        f_fac::Matrix{Float64}, 
        ::Lazy,
        s::Union{Matrix{Float64},Nothing}) where {d}

        to = get_timer(string("thread_timer_", Threads.threadid()))

        @timeit to "volume terms" begin
            volume_terms = zero(residual)
            @inbounds for m in 1:d
                volume_terms += mul!(residual, operators.VOL[m], f[m])
            end
        end

        @timeit to "facet terms" begin
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end
 
        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit to "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        @timeit to "mass matrix solve" begin
            residual = operators.M \ rhs
        end
        
        return residual
    end

    """
        Auxiliary variable in physical-operator form
    """
    function auxiliary_variable!(m::Int, 
        q::Matrix{Float64},
        operators::PhysicalOperators{d},
        u::Matrix{Float64},
        u_fac::Matrix{Float64}, 
        ::Eager) where {d}

        to = get_timer(string("thread_timer_", Threads.threadid()))

        @timeit to "volume terms" begin
            volume_terms = -1.0*mul!(q, operators.VOL[m], u)
        end

        @timeit to "facet terms" begin
            facet_terms = -1.0*mul!(q, operators.FAC, u_fac)
        end

        q = volume_terms + facet_terms
        return q
    end


    """
    Auxiliary variable in reference-operator form
    """
    function auxiliary_variable!(m::Int, 
        q::Matrix{Float64},
        operators::PhysicalOperators{d},
        u::Matrix{Float64},
        u_fac::Matrix{Float64}, 
        ::Lazy) where {d}

        @unpack VOL, FAC, M = operators
        to = get_timer(string("thread_timer_", Threads.threadid()))

        @timeit to "volume terms" begin
            volume_terms = -1.0*mul!(q, VOL[m], u)
        end

        @timeit to "facet terms" begin
            facet_terms = -1.0*mul!(q, FAC, u_fac)
        end
        
        @timeit to "mass matrix solve" begin
            q = M \ (volume_terms + facet_terms)
        end
        return q
    end

#TODO add flux diff.
#=
    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},
        F::NTuple{d,Array{Float64,3}}, 
        f_fac::Matrix{Float64}, 
        ::Lazy,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}
        to = get_timer(string("thread_timer_", Threads.threadid()))

        @timeit to "volume terms" begin
            volume_terms = zero(residual)
            @inbounds for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit to "facet terms" begin
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms
 
        if !isnothing(s)
            @timeit to "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        @timeit to "mass matrix solve" begin
            residual = operators.M \ rhs
        end
        
        return residual
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},
        F::NTuple{d,Array{Float64,3}}, 
        f_fac::Matrix{Float64}, 
        ::Eager,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}
        to = get_timer(string("thread_timer_", Threads.threadid()))

        @timeit to "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit to "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit to "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        return rhs
        return residual
    end
=#
    export StrongConservationForm, WeakConservationForm
    include("conservation_form.jl")

    export LinearResidual
    include("linear.jl")

end