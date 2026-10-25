using StableSpectralElements, BenchmarkTools, LinearAlgebra, SparseArrays, Printf
using TimerOutputs: disable_timer!, enable_timer!

function benchmark_sparse_operators(p = 4, elements_per_direction = 2)
    reference = ReferenceApproximation(NodalTensor(p), Tri())
    mesh = uniform_periodic_mesh(
        reference,
        ((0.0, 1.0), (0.0, 1.0)),
        (elements_per_direction, elements_per_direction),
    )
    spatial = SpatialDiscretization(mesh, reference)
    equation = LinearAdvectionEquation((1.0, 1.0))
    initial_data = InitialDataSine(1.0, (2π, 2π))
    problems = Tuple(
        semidiscretize(
            equation,
            spatial,
            initial_data,
            StandardForm(),
            (0.0, 0.1),
            ReferenceOperator(),
            algorithm,
            parallelism = Serial(),
        ) for algorithm in (DefaultOperatorAlgorithm(), SparseMatrixAlgorithm())
    )

    println("Julia $VERSION, degree $p, $(spatial.N_e) triangles, serial residual")
    println("Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads())")
    println("Median timings; operators use four solution components; setup is excluded.")
    println("Flux-differencing S and C operators are not part of this comparison.")

    # Apply identical reference operators and facet maps in both representations
    for name in (:D, :Dᵀ, :R, :Rᵀ)
        for (label, problem) in zip(("default", "sparse"), problems)
            operators = problem.p.operators
            operator =
                name in (:D, :Dᵀ) ? getproperty(operators, name)[1] :
                getproperty(operators, name)
            x = reshape(sin.(1.0:(4*size(operator, 2))), size(operator, 2), 4)
            y = zeros(size(operator, 1), 4)
            trial = @benchmark mul!($y, $operator, $x) samples = 1000 seconds = 1 evals = 1
            estimate = median(trial)
            @printf(
                "%-3s %-7s size=%s nnz=%d time=%.3f μs bytes=%d allocs=%d\n",
                String(name),
                label,
                string(size(operator)),
                nnz(sparse(Matrix(operator))),
                estimate.time / 1.0e3,
                estimate.memory,
                estimate.allocs
            )
        end
    end

    # Disable the same TimerOutputs instrumentation for both residuals
    disable_timer!()
    for (label, problem) in zip(("default", "sparse"), problems)
        rhs = similar(problem.u0)
        f, u, solver = problem.f, problem.u0, problem.p
        trial = @benchmark $f($rhs, $u, $solver, 0.0) samples = 1000 seconds = 1 evals = 1
        estimate = median(trial)
        @printf(
            "residual %-7s time=%.3f μs bytes=%d allocs=%d\n",
            label,
            estimate.time / 1.0e3,
            estimate.memory,
            estimate.allocs
        )
    end
    enable_timer!()
    return nothing
end

benchmark_sparse_operators()
