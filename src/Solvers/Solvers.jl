module Solvers

    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, inv, mul!
    using TimerOutputs
    using LinearMaps: LinearMap

    using ..ConservationLaws: ConservationLaw, physical_flux, numerical_flux, two_point_flux
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization
    using ..ParametrizedFunctions: AbstractParametrizedFunction, AbstractParametrizedFunction, evaluate
    using ..Operators: flux_diff

    export AbstractResidualForm, AbstractPhysicalOperators, AbstractMappingForm, AbstractCouplingForm, AbstractStrategy, PhysicaOperators, Eager, Lazy, Solver, StandardMapping, SkewSymmetricMapping, CreanMapping, StandardCoupling, SkewSymmetricCoupling, initialize, semidiscretize, precompute, apply_operators!, combine, get_dof, rhs!

    abstract type AbstractResidualForm end
    abstract type AbstractMappingForm end
    abstract type AbstractCouplingForm end
    abstract type AbstractPhysicalOperators{d} end
    abstract type AbstractStrategy end

    struct Eager <: AbstractStrategy end
    struct Lazy <: AbstractStrategy end
    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end
    struct CreanMapping <: AbstractMappingForm end

    struct Solver{ResidualForm,PhysicalOperators,d,N_eq}
        conservation_law::ConservationLaw{d,N_eq}
        operators::Vector{PhysicalOperators}
        x_q::NTuple{d,Matrix{Float64}}
        connectivity::Matrix{Int}
        form::ResidualForm
        strategy::AbstractStrategy
    end

    struct PhysicalOperators{d} <: AbstractPhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        SRC::LinearMap
        M::AbstractMatrix
        V::LinearMap
        Vf::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}} 
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    function Solver(conservation_law::ConservationLaw,     
        spatial_discretization::SpatialDiscretization,
        form::AbstractResidualForm,
        strategy::Lazy)
        return Solver(conservation_law, 
            make_operators(spatial_discretization, form),
            spatial_discretization.mesh.xyzq,
            spatial_discretization.mesh.mapP, form, strategy)
    end

    function Solver(conservation_law::ConservationLaw,     
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
        ::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

        @unpack N_el, M, geometric_factors = spatial_discretization
        @unpack N_p, N_q, V, W = spatial_discretization.reference_approximation
        @unpack xyzq = spatial_discretization.mesh

        u0 = Array{Float64}(undef, N_p, N_eq, N_el)
        for k in 1:N_el
            # project to solution DOF
            u0[:,:,k] = M[k] \ convert(Matrix, V' * W * 
                Diagonal(geometric_factors.J_q[:,k]) * 
                evaluate(initial_data, Tuple(xyzq[m][:,k] for m in 1:d)))
        end
        return u0
    end

    function semidiscretize(
        conservation_law::ConservationLaw,spatial_discretization::SpatialDiscretization,
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
            M, V, Vf, NTR, scaled_normal)
    end

    function combine(operator::LinearMap)
        return LinearMap(convert(Matrix,operator))
    end

    function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        ::ConservationLaw{d,N_eq}) where {d, N_eq}
        return (spatial_discretization.reference_approximation.N_p, 
            N_eq, spatial_discretization.N_el)
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},  
        f::NTuple{d,Matrix{Float64}}, 
        f_fac::Matrix{Float64}, ::Eager,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}
        
        @timeit "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += mul!(residual, operators.VOL[m], f[m])
            end
        end

        @timeit "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        # add together
        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        return rhs
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},
        f::NTuple{d,Matrix{Float64}}, 
        f_fac::Matrix{Float64}, 
        ::Lazy,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}

        @timeit "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += mul!(residual, operators.VOL[m], f[m])
            end
        end

        @timeit "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms
 
        if !isnothing(s)
            @timeit "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        @timeit "mass matrix solve" begin
            # add together and solve
            residual = operators.M \ rhs
        end
        
        return residual
    end

    function apply_operators!(residual::Matrix{Float64},
        operators::PhysicalOperators{d},
        F::NTuple{d,Array{Float64,3}}, 
        f_fac::Matrix{Float64}, 
        ::Lazy,
        s::Union{Matrix{Float64},Nothing}=nothing) where {d}

        @timeit "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        rhs = volume_terms + facet_terms
 
        if !isnothing(s)
            @timeit "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        @timeit "mass matrix solve" begin
            # add together and solve
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

        @timeit "volume terms" begin
            # compute volume terms
            volume_terms = zero(residual)
            for m in 1:d
                volume_terms += flux_diff(operators.VOL[m], F[m])
            end
        end

        @timeit "facet terms" begin
            # compute facet terms
            facet_terms = mul!(residual, operators.FAC, f_fac)
        end

        # add together
        rhs = volume_terms + facet_terms

        if !isnothing(s)
            @timeit "source terms" begin
                source_terms = mul!(residual, operators.SRC, s)
            end
            rhs = rhs + source_terms
        end

        return rhs
        return residual
    end

    export StrongConservationForm, WeakConservationForm, MixedConservationForm
    include("conservation_form.jl")

    export StrongFluxDiffForm
    include("flux_diff_form.jl")

    export LinearResidual
    include("linear.jl")

end