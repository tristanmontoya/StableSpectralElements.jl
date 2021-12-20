module InitialConditions

    using ..ConservationLaws: ConservationLaw
    using ..SpatialDiscretizations: SpatialDiscretization

    export AbstractInitialData, InitialDataSine, evaluate
    
    abstract type AbstractInitialData{d} end

    struct InitialDataSine{d} <: AbstractInitialData{d}
        A::Float64  # amplitude
        k::NTuple{d,Float64}  # wave number in each direction
        N_eq::Int
    end

    function InitialDataSine(A::Float64, k::Float64; N_eq::Int=1)
        return InitialDataSine(A,(k,),N_eq)
    end

    function InitialDataSine(A::Float64, k::NTuple{d,Float64}; 
        N_eq::Int=1) where {d}
        return InitialDataSine(A,k,N_eq)
    end

    function evaluate(initial_data::InitialDataSine{d}, 
        x::NTuple{d,Float64}) where {d}
        return fill(initial_data.A*prod(Tuple(sin(initial_data.k[m]*x[m])
            for m in 1:d)), initial_data.N_eq)
    end

    function evaluate(initial_data::AbstractInitialData{d},
        x::NTuple{d,Vector{Float64}}) where {d}
        N = length(x[1])
        u0 = Matrix{Float64}(undef, N, initial_data.N_eq)
        for i in 1:N
            u0[i,:] = evaluate(initial_data, Tuple(x[m][i] for m in 1:d))
        end
        return u0
    end

    function evaluate(initial_data::AbstractInitialData{d},
        x::NTuple{d,Matrix{Float64}}) where {d}
        N, N_el = size(x[1])
        u0 = Array{Float64}(undef, N, initial_data.N_eq, N_el)
        for k in 1:N_el
            u0[:,:,k] = evaluate(initial_data, Tuple(x[m][:,k] for m in 1:d))
        end
        return u0
    end

    
end