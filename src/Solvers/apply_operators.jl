function apply_operators!( 
    rhs::AbstractMatrix{Float64}, 
    tmp::AbstractMatrix{Float64},
    operators::DiscretizationOperators{d},
    f::NTuple{d,Matrix{Float64}}, 
    f_fac::Matrix{Float64}, 
    s::Union{Matrix{Float64},Nothing}=nothing) where {d}

    rhs_q = zero(tmp)

    @timeit thread_timer() "volume terms" @inbounds for m in 1:d
        mul!(tmp,  operators.VOL[m], f[m])
        rhs_q .+= tmp
    end

    @timeit thread_timer() "facet terms" begin
        mul!(tmp,  operators.FAC, f_fac)
        rhs_q .+= tmp
    end

    isnothing(s) || @timeit thread_timer() "source terms" begin
        mul!(tmp,  operators.SRC, s)
        rhs_q .+= tmp
    end

    @timeit thread_timer() "mul test function"  mul!(rhs, operators.V', rhs_q)
    @timeit thread_timer() "mass matrix solve" ldiv!(operators.M, rhs)
    return rhs
end

function auxiliary_variable(m::Int, 
    operators::DiscretizationOperators{d},
    u::Matrix{Float64},
    u_fac::Matrix{Float64}) where {d}

    @unpack VOL, FAC, M, V = operators
    N_q = size(u,1)
    N_c = size(u,2)
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

    @timeit thread_timer() "mass matrix solve" ldiv!(M, rhs)
    return rhs
end