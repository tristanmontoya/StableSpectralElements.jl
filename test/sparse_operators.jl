using LinearAlgebra: Diagonal, mul!
using SparseArrays: SparseMatrixCSC, sparse

@testset "Sparse reference operators" begin
    selection = SelectionMap([1, 3], 4)
    @test make_operator(selection, SparseMatrixAlgorithm()) === selection
    @test make_operator(Diagonal([1.0, 2.0]), SparseMatrixAlgorithm()) ==
          sparse(Diagonal([1.0, 2.0]))

    for approx_type in (NodalTensor(3), ModalTensor(2))
        reference = ReferenceApproximation(approx_type, Tri())
        mesh = uniform_periodic_mesh(reference, ((0.0, 1.0), (0.0, 1.0)), (2, 2))
        spatial = SpatialDiscretization(mesh, reference)
        advection = LinearAdvectionEquation((1.0, 1.0))
        initial_data = InitialDataSine(1.0, (2π, 2π))
        default_problem = semidiscretize(
            advection,
            spatial,
            initial_data,
            StandardForm(),
            (0.0, 0.1),
            ReferenceOperator(),
            DefaultOperatorAlgorithm(),
            parallelism = Serial(),
        )
        sparse_problem = semidiscretize(
            advection,
            spatial,
            initial_data,
            StandardForm(),
            (0.0, 0.1),
            ReferenceOperator(),
            SparseMatrixAlgorithm(),
            parallelism = Serial(),
        )
        default_ops = default_problem.p.operators
        sparse_ops = sparse_problem.p.operators

        # Compare forward and transpose application with several solution components
        for (default_op, sparse_op) in
            zip((default_ops.D..., default_ops.R), (sparse_ops.D..., sparse_ops.R))
            @test sparse_op isa SparseMatrixCSC
            x = reshape(sin.(1.0:(3*size(default_op, 2))), size(default_op, 2), 3)
            y_default = zeros(size(default_op, 1), 3)
            y_sparse = similar(y_default)
            mul!(y_default, default_op, x)
            mul!(y_sparse, sparse_op, x)
            @test y_sparse ≈ y_default atol = 1.0e-12

            x_transpose = copy(y_default)
            y_default = zeros(size(default_op, 2), 3)
            y_sparse = similar(y_default)
            mul!(y_default, transpose(default_op), x_transpose)
            mul!(y_sparse, transpose(sparse_op), x_transpose)
            @test y_sparse ≈ y_default atol = 1.0e-12
        end

        # The residual includes sparse differentiation, facet maps, and their transposes
        rhs_default = similar(default_problem.u0)
        rhs_sparse = similar(rhs_default)
        default_problem.f(rhs_default, default_problem.u0, default_problem.p, 0.0)
        sparse_problem.f(rhs_sparse, sparse_problem.u0, sparse_problem.p, 0.0)
        @test rhs_sparse ≈ rhs_default atol = 1.0e-11

        # Exercise both nodal and modal entropy projection with a non-selection facet map
        euler = EulerEquations{2}()
        euler_data = EulerPeriodicTest(euler)
        default_euler = semidiscretize(
            euler,
            spatial,
            euler_data,
            FluxDifferencingForm(),
            (0.0, 0.1),
            ReferenceOperator(),
            DefaultOperatorAlgorithm(),
            parallelism = Serial(),
        )
        sparse_euler = semidiscretize(
            euler,
            spatial,
            euler_data,
            FluxDifferencingForm(),
            (0.0, 0.1),
            ReferenceOperator(),
            SparseMatrixAlgorithm(),
            parallelism = Serial(),
        )
        rhs_default = similar(default_euler.u0)
        rhs_sparse = similar(rhs_default)
        default_euler.f(rhs_default, default_euler.u0, default_euler.p, 0.0)
        sparse_euler.f(rhs_sparse, sparse_euler.u0, sparse_euler.p, 0.0)
        @test rhs_sparse ≈ rhs_default atol = 1.0e-11

        if approx_type isa ModalTensor
            # Sparse differentiation retains the modal interpolation representation
            @test typeof(sparse_ops.V) == typeof(default_ops.V)
            for orthonormal in (Val(true), Val(false))
                default_mass = WeightAdjustedSolver(
                    spatial.geometric_factors.J_q,
                    reference.V,
                    reference.W,
                    DefaultOperatorAlgorithm(),
                    orthonormal,
                    1.0e-12,
                )
                sparse_mass = WeightAdjustedSolver(
                    spatial.geometric_factors.J_q,
                    reference.V,
                    reference.W,
                    SparseMatrixAlgorithm(),
                    orthonormal,
                    1.0e-12,
                )
                @test typeof(sparse_mass.V) == typeof(default_mass.V)
                @test Matrix(sparse_mass.V) ≈ Matrix(default_mass.V)
                @test Matrix(sparse_mass.Vᵀ) ≈ Matrix(default_mass.Vᵀ)
            end

            # Existing BLAS and generic choices continue to control interpolation
            for algorithm in (BLASAlgorithm(), GenericMatrixAlgorithm())
                problem = semidiscretize(
                    advection,
                    spatial,
                    initial_data,
                    StandardForm(),
                    (0.0, 0.1),
                    ReferenceOperator(),
                    algorithm,
                    parallelism = Serial(),
                )
                @test typeof(problem.p.operators.V) ==
                      typeof(make_operator(reference.V, algorithm))
            end
        end
    end
end
