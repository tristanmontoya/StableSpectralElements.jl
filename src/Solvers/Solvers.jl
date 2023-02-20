module Solvers

    using UnPack
    import LinearAlgebra
    using LinearAlgebra: Diagonal, eigvals, inv, mul!, factorize, cholesky,ldiv!, Factorization, Cholesky, Symmetric
    using TimerOutputs
    using LinearMaps: LinearMap, UniformScalingMap
    using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEqAlgorithm, solve
    
    using ..MatrixFreeOperators: AbstractOperatorAlgorithm, BLASAlgorithm, GenericMatrixAlgorithm, DefaultOperatorAlgorithm, WeightAdjustedMap, make_operator
    using ..ConservationLaws: AbstractConservationLaw, AbstractPDEType, FirstOrder, SecondOrder, AbstractInviscidNumericalFlux, AbstractViscousNumericalFlux, AbstractTwoPointFlux, NoInviscidFlux, NoViscousFlux, NoTwoPointFlux, NoSourceTerm, physical_flux, numerical_flux, LaxFriedrichsNumericalFlux, BR1
    using ..SpatialDiscretizations: ReferenceApproximation, SpatialDiscretization, check_facet_nodes, check_normals
    using ..GridFunctions: AbstractGridFunction, AbstractGridFunction, NoSourceTerm, evaluate   
    
    export AbstractResidualForm, AbstractMappingForm, AbstractStrategy, DiscretizationOperators, PhysicalOperator, ReferenceOperator, Solver, StandardMapping, SkewSymmetricMapping, get_dof, CLOUD_print_timer, CLOUD_reset_timer!, thread_timer, rhs!, make_operators

    abstract type AbstractResidualForm{MappingForm, TwoPointFlux} end
    abstract type AbstractMappingForm end
    abstract type AbstractStrategy end

    struct StandardMapping <: AbstractMappingForm end
    struct SkewSymmetricMapping <: AbstractMappingForm end
    struct PhysicalOperator <: AbstractStrategy end
    struct ReferenceOperator <: AbstractStrategy end

    struct DiscretizationOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        SRC::LinearMap
        M::Union{Cholesky, AbstractMatrix, WeightAdjustedMap}
        V::LinearMap
        Vf::LinearMap
        n_f::NTuple{d, Vector{Float64}}
        N_p::Int
        N_q::Int
        N_f::Int
    end

    struct Solver{d,ResidualForm,PDEType}
        conservation_law::AbstractConservationLaw{d,PDEType}
        operators::Vector{<:DiscretizationOperators}
        x_q::NTuple{d,Matrix{Float64}}
        connectivity::Matrix{Int}
        form::ResidualForm
        N_e::Int
        
        function Solver(conservation_law::AbstractConservationLaw{d,PDEType},
            operators::Vector{<:DiscretizationOperators},
            x_q::NTuple{d,Matrix{Float64}},
            connectivity::Matrix{Int},
            form::ResidualForm) where {d,ResidualForm,PDEType}
            return new{d,ResidualForm,PDEType}(conservation_law, operators,
                x_q, connectivity, form, length(operators))
        end
    end

    function get_dof(spatial_discretization::SpatialDiscretization{d}, 
        conservation_law::AbstractConservationLaw{d}) where {d}
        return (spatial_discretization.reference_approximation.N_p, 
            conservation_law.N_c, spatial_discretization.N_e)
    end

    function CLOUD_reset_timer!()
        for t in 1:Threads.nthreads()
            thread_timer = get_timer(string("thread_timer_",t))
            reset_timer!(thread_timer)
        end
    end
    
    function CLOUD_print_timer()
        for t in 1:Threads.nthreads()
            thread_timer = get_timer(string("thread_timer_",t))
            print_timer(thread_timer, title=string("Thread ", t))
        end
    end

    function thread_timer()
        return get_timer(string("thread_timer_", Threads.threadid()))
    end
    
    
    export AbstractMassMatrixSolver, CholeskySolver, WeightAdjustedSolver, mass_matrix
    include("mass_matrix.jl") 

    export initialize, semidiscretize, precompute
    include("preprocessing.jl")

    export apply_operators!, auxiliary_variable
    include("apply_operators.jl")

    export StrongConservationForm, WeakConservationForm
    include("conservation_form.jl")

    export LinearResidual
    include("linear.jl")
end