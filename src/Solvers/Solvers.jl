module Solvers

    using OrdinaryDiffEq: ODEProblem
    using UnPack
    using LinearAlgebra: Diagonal, inv
    using LinearMaps: LinearMap
    using TimerOutputs

    using ..ConservationLaws: ConservationLaw, physical_flux, numerical_flux
    using ..SpatialDiscretizations: SpatialDiscretization
    using ..InitialConditions: AbstractInitialData, initial_condition
    
    export AbstractResidualForm, AbstractPhysicalOperators, AbstractStrategy, Solver, PhysicalOperatorsEager, PhysicalOperatorsEager, Eager, Lazy, initialize, semidiscretize, apply_operators, rhs!

    abstract type AbstractResidualForm end
    abstract type AbstractPhysicalOperators{d} end
    abstract type AbstractStrategy end

    struct Eager <: AbstractStrategy end
    struct Lazy <: AbstractStrategy end
    struct Solver{ResidualForm,PhysicalOperators,d,N_eq}
        conservation_law::ConservationLaw{d,N_eq}
        operators::Vector{PhysicalOperators}
        connectivity::Matrix{Int}
        form::ResidualForm
    end

    struct PhysicalOperatorsEager{d} <: AbstractPhysicalOperators{d}
        VOL::NTuple{d,LinearMap}
        FAC::LinearMap
        R::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}}  # only needed for strong form
        scaled_normal::NTuple{d, Vector{Float64}}
    end

    struct PhysicalOperatorsLazy{d} <: AbstractPhysicalOperators{d}
        vol::NTuple{d,LinearMap}  # not pre-multipled by mass inverse
        fac::LinearMap  # not pre-multiplied by mass inverse
        M::AbstractMatrix
        R::LinearMap
        NTR::NTuple{d,Union{LinearMap,AbstractMatrix}}  # only needed for strong form
        scaled_normal::NTuple{d, Vector}
    end
    
    function initialize(initial_data::AbstractInitialData,
        conservation_law::ConservationLaw{d,N_eq},
        spatial_discretization::SpatialDiscretization{d}) where {d, N_eq}

        @unpack N_el, M, geometric_factors = spatial_discretization
        @unpack N_p, N_q, V, W = spatial_discretization.reference_approximation
        @unpack xyzq = spatial_discretization.mesh

        f = initial_condition(initial_data, conservation_law)
        
        u0 = Array{Float64}(undef, N_p, N_eq, N_el)
        u0q = Matrix{Float64}(undef, N_q, N_eq)
        for k in 1:N_el
        
            #evaluate initial condition at each node
            for i in 1:N_q
                u0q[i,:] = f(Tuple(xyzq[m][i,k] for m in 1:d)) 
            end
            
            # project to solution DOF
            u0[:,:,k] = M[k] \ convert(Matrix, transpose(V) * W * 
                Diagonal(geometric_factors.J[:,k]) * u0q)
        end
        return u0
    end

    function apply_operators(operators::PhysicalOperatorsEager{d},
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}) where {d}
        return convert(Matrix, sum(operators.VOL[m] * f[m] for m in 1:d) + 
            operators.FAC * f_fac)
    end

    function apply_operators(operators::PhysicalOperatorsLazy{d},
        f::NTuple{d,Matrix{Float64}}, f_fac::Matrix{Float64}) where {d}
        return operators.M \ convert(Matrix, sum(operators.vol[m] * f[m] for m in 1:d) + operators.fac * f_fac)
    end

    # utils
    function combine(operator::LinearMap)
        return LinearMap(convert(Matrix,operator))
    end

    export StrongConservationForm
    include("strong_conservation_form.jl")

end