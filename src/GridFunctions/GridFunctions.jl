module GridFunctions

using StaticArrays

export AbstractGridFunction,
       ConstantFunction,
       InitialDataSine,
       InitialDataCosine,
       InitialDataGaussian,
       InitialDataSinCos,
       DerivativeSinCos,
       InitialDataGassner,
       BurgersSolution,
       SourceTermGassner,
       GaussianNoise,
       NoSourceTerm,
       evaluate

abstract type AbstractGridFunction{d} end

struct ConstantFunction{d} <: AbstractGridFunction{d}
    c::Float64
    N_c::Int

    function ConstantFunction{d}(c) where {d}
        return new{d}(c, 1)
    end
end

struct InitialDataSine{d} <: AbstractGridFunction{d}
    A::Float64  # amplitude
    k::NTuple{d, Float64}  # wave number in each direction
    N_c::Int

    function InitialDataSine(A::Float64, k::NTuple{d, Float64}) where {d}
        return new{d}(A, k, 1)
    end
end

struct InitialDataCosine{d} <: AbstractGridFunction{d}
    A::Float64  # amplitude
    k::NTuple{d, Float64}  # wave number in each direction
    N_c::Int

    function InitialDataCosine(A::Float64, k::NTuple{d, Float64}) where {d}
        return new{d}(A, k, 1)
    end
end

struct InitialDataGaussian{d} <: AbstractGridFunction{d}
    A::Float64  # amplitude
    σ::Float64 # width
    x₀::NTuple{d, Float64}
    N_c::Int
    function InitialDataGaussian(A::Float64, σ::Float64, x₀::NTuple{d, Float64}) where {d}
        return new{d}(A, σ, x₀, 1)
    end
end

struct InitialDataGassner <: AbstractGridFunction{1}
    k::Float64
    ϵ::Float64
    N_c::Int

    function InitialDataGassner(k, ϵ)
        return new(k, ϵ, 1)
    end
end

struct SourceTermGassner <: AbstractGridFunction{1}
    k::Float64
    ϵ::Float64
    N_c::Int

    function SourceTermGassner(k, ϵ)
        return new(k, ϵ, 1)
    end
end

struct InitialDataSinCos <: AbstractGridFunction{2}
    A::Float64  # amplitude
    k::NTuple{2, Float64}  # wave number in each direction
    N_c::Int

    function InitialDataSinCos(A::Float64, k::NTuple{2, Float64})
        return new(A, k, 1)
    end
end

struct DerivativeSinCos <: AbstractGridFunction{2}
    A::Float64  # amplitude
    k::NTuple{2, Float64}  # wave number in each direction
    N_c::Int

    function DerivativeSinCos(A::Float64, k::NTuple{2, Float64})
        return new(A, k, 1)
    end
end
struct GaussianNoise{d} <: AbstractGridFunction{d}
    σ::Float64
    N_c::Int
end

struct NoSourceTerm{d} <: AbstractGridFunction{d} end

@inline function InitialDataSine(A::Float64, k::Float64)
    return InitialDataSine(A, (k,))
end

@inline function InitialDataCosine(A::Float64, k::Float64)
    return InitialDataCosine(A, (k,))
end

@inline function InitialDataGaussian(A::Float64, σ::Float64, x₀::Float64)
    return InitialDataGaussian(A, σ, (x₀,))
end

@inline function evaluate(f::ConstantFunction{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    return fill(f.c, f.N_c)
end

@inline function evaluate(f::InitialDataSine{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    return fill(f.A * prod(Tuple(sin(f.k[m] * x[m]) for m in 1:d)), f.N_c)
end

@inline function evaluate(f::InitialDataCosine{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    return fill(f.A * prod(Tuple(cos(f.k[m] * x[m]) for m in 1:d)), f.N_c)
end

@inline function evaluate(f::InitialDataGaussian{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    (; A, σ, x₀, N_c) = f
    r² = sum((x[m] - x₀[m]) .^ 2 for m in 1:d)
    return fill(A * exp.(-r² / (2.0 * σ^2)), N_c)
end

@inline function evaluate(f::InitialDataGassner, x::NTuple{1, Float64}, t::Float64 = 0.0)
    return [sin(f.k * x[1]) + f.ϵ]
end

@inline function evaluate(f::SourceTermGassner, x::NTuple{1, Float64}, t::Float64 = 0.0)
    return SVector{1}(f.k .* cos(f.k * (x[1] - t)) * (-1.0 + f.ϵ + sin(f.k * (x[1] - t))))
end

@inline function evaluate(f::GaussianNoise{d},
        x::NTuple{d, Float64},
        t::Float64 = 0.0) where {d}
    return [f.σ * randn() for e in 1:(f.N_c)]
end

@inline function evaluate(f::AbstractGridFunction{d},
        x::NTuple{d, AbstractVector{Float64}},
        t::Float64 = 0.0) where {d}
    N = length(x[1])
    u0 = Matrix{Float64}(undef, N, f.N_c)
    @inbounds for i in 1:N
        u0[i, :] .= evaluate(f, Tuple(x[m][i] for m in 1:d), t)
    end
    return u0
end

function evaluate(f::AbstractGridFunction{d},
        x::NTuple{d, AbstractMatrix{Float64}},
        t::Float64 = 0.0) where {d}
    N, N_e = size(x[1])
    u0 = Array{Float64}(undef, N, f.N_c, N_e)
    @inbounds for k in 1:N_e
        u0[:, :, k] .= evaluate(f, Tuple(x[m][:, k] for m in 1:d), t)
    end
    return u0
end

function evaluate(f::Function,
        x::NTuple{d, AbstractMatrix{Float64}},
        t::Float64 = 0.0) where {d}
    u0 = Array{Float64}(undef,
        size(x[1], 1),
        length(f(Tuple(0.0 for m in 1:d)..., t)),
        size(x[1], 2))
    @inbounds for k in axes(x[1], 2), i in axes(x[1], 1)
        u0[i, :, k] .= f(Tuple(x[m][i, k] for m in 1:d)..., t)
    end
    return u0
end

function evaluate(f::Function,
        x::NTuple{d, AbstractVector{Float64}},
        t::Float64 = 0.0) where {d}
    u0 = Matrix{Float64}(undef, length(x[1]), length(f(Tuple(0.0 for m in 1:d)..., t)))
    @inbounds for i in eachindex(x[1])
        u0[i, :] .= f(Tuple(x[m][i] for m in 1:d)..., t)
    end
    return u0
end

@inline function evaluate(::NoSourceTerm{d},
        ::NTuple{d, Vector{Float64}},
        ::Float64) where {d}
    return nothing
end

@inline function evaluate(f::InitialDataSinCos, x::NTuple{2, Float64}, t::Float64 = 0.0)
    return f.A * sin(f.k[1] * x[1]) * cos(f.k[2] * x[2])
end

@inline function evaluate(f::DerivativeSinCos, x::NTuple{2, Float64}, t::Float64 = 0.0)
    return f.A * f.k[1] * cos(f.k[1] * x[1]) * cos(f.k[2] * x[2])
end
end
