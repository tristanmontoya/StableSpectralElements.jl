using BenchmarkTools, LinearAlgebra, Printf, StableSpectralElements

# Direct formulas provide a baseline for the logarithmic conversions
@inline function baseline_conservative_to_entropy!(
    w,
    conservation_law::EulerEquations{d},
    u,
) where {d}
    (; γ, γ_minus_1, inv_γ_minus_1) = conservation_law
    k = (0.5 / u[1]) * sum(u[m+1]^2 for m = 1:d)
    p = γ_minus_1 * (u[end] - k)
    inv_p = 1.0 / p
    w[1] = inv_γ_minus_1 * (γ - log(p / u[1]^γ)) - k * inv_p
    @inbounds for m = 1:d
        w[m+1] = u[m+1] * inv_p
    end
    w[d+2] = -u[1] * inv_p
    return
end

@inline function baseline_entropy_to_conservative!(
    u,
    conservation_law::EulerEquations{d},
    w,
) where {d}
    (; γ, γ_minus_1, inv_γ_minus_1) = conservation_law
    w = w * γ_minus_1
    k = sum(w[m+1]^2 for m = 1:d) / (2 * w[d+2])
    s = γ - w[1] + k
    ρe = (γ_minus_1 / (-w[d+2])^γ)^inv_γ_minus_1 * exp(-s * inv_γ_minus_1)
    u[1] = -w[d+2] * ρe
    @inbounds for m = 1:d
        u[m+1] = w[m+1] * ρe
    end
    u[d+2] = ρe * (1 - k)
    return
end

function benchmark_entropy_conversions()
    println(
        "Julia ",
        VERSION,
        "; CPU: ",
        Sys.CPU_NAME,
        "; Julia threads: ",
        Threads.nthreads(),
        "; BLAS threads: ",
        BLAS.get_num_threads(),
    )
    println("dimension,layout,direction,implementation,median_ns,bytes,allocations")

    for d = 1:3, row_view in (false, true)
        conservation_law = EulerEquations{d}(1.4)
        velocity = [(-1.0)^m * (0.2 + 0.1 * m) for m = 1:d]
        state = [1.2; 1.2 .* velocity; 0.9 / 0.4 + 0.6 * sum(abs2, velocity)]
        u = row_view ? view(repeat(reshape(state, 1, :), 16, 1), 8, :) : copy(state)
        w = row_view ? view(zeros(16, d + 2), 8, :) : zeros(d + 2)
        reference_w = similar(state)
        recovered = row_view ? view(zeros(16, d + 2), 8, :) : similar(state)
        reference_u = similar(state)

        baseline_conservative_to_entropy!(reference_w, conservation_law, u)
        conservative_to_entropy!(w, conservation_law, u)
        @assert isapprox(w, reference_w; rtol = 1.0e-13, atol = 1.0e-13)
        baseline_entropy_to_conservative!(reference_u, conservation_law, w)
        entropy_to_conservative!(recovered, conservation_law, w)
        @assert isapprox(recovered, reference_u; rtol = 1.0e-13, atol = 1.0e-13)

        forward_baseline =
            @benchmark baseline_conservative_to_entropy!($w, $conservation_law, $u) seconds =
                0.5
        forward =
            @benchmark conservative_to_entropy!($w, $conservation_law, $u) seconds = 0.5
        inverse_baseline =
            @benchmark baseline_entropy_to_conservative!($recovered, $conservation_law, $w) seconds =
                0.5
        inverse =
            @benchmark entropy_to_conservative!($recovered, $conservation_law, $w) seconds =
                0.5

        for (direction, implementation, trial) in (
            ("forward", "baseline", forward_baseline),
            ("forward", "logarithmic", forward),
            ("inverse", "baseline", inverse_baseline),
            ("inverse", "logarithmic", inverse),
        )
            estimate = median(trial)
            @printf(
                "%d,%s,%s,%s,%.2f,%d,%d\n",
                d,
                row_view ? "row_view" : "vector",
                direction,
                implementation,
                estimate.time,
                estimate.memory,
                estimate.allocs
            )
        end
    end
end

benchmark_entropy_conversions()
