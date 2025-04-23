
# This removes the threading and considers just one element. Note that this probably will 
# not work with the Euler equations as the facet states at adjacent elements are left 
# undefined and thus may lead to non-physical states when used to compute the fluxes.
@views @timeit "semi-disc. residual" function rhs_benchmark!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{d,
                FirstOrder},
            <:ReferenceOperators,
            <:AbstractMassMatrixSolver,
            <:StandardForm},
        t::Float64 = 0.0) where {d}
    @timeit "unpack" begin
        (; conservation_law, connectivity, form) = solver
        (; inviscid_numerical_flux) = form
        (; f_q, f_f, f_n, u_q, r_q, u_f, temp, CI) = solver.preallocated_arrays
        (; D, V, R, halfWΛ, halfN, BJf, n_f) = solver.operators
    end

    k = 1  #just one element

    @timeit "vandermonde" mul!(u_q[:, :, k], V, u[:, :, k])

    @timeit "extrap solution" mul!(u_f[:, k, :], R, u_q[:, :, k])

    @timeit "phys flux" physical_flux!(f_q[:, :, :, k], conservation_law, u_q[:, :, k])

    @timeit "num flux" numerical_flux!(f_f[:, :, k],
        conservation_law,
        inviscid_numerical_flux,
        u_f[:, k, :],
        u_f[CI[connectivity[:, k]], :],
        n_f[:, :, k])

    @timeit "fill w zeros" fill!(r_q[:, :, k], 0.0)

    @inbounds for n in 1:d
        @inbounds @timeit "volume operators" for m in 1:d
            mul!(temp[:, :, k], halfWΛ[m, n, k], f_q[:, :, n, k])
            mul!(u_q[:, :, k], D[m]', temp[:, :, k])
            r_q[:, :, k] .+= u_q[:, :, k]
            mul!(u_q[:, :, k], D[m], f_q[:, :, n, k])
            lmul!(halfWΛ[m, n, k], u_q[:, :, k])
            r_q[:, :, k] .-= u_q[:, :, k]
        end

        # difference facet flux
        @timeit "diff flux" begin
            mul!(f_n[:, :, k], R, f_q[:, :, n, k])
            lmul!(halfN[n, k], f_n[:, :, k])
            f_f[:, :, k] .-= f_n[:, :, k]
        end
    end

    # apply facet operators
    @timeit "facet operators" begin
        lmul!(BJf[k], f_f[:, :, k])
        mul!(u_q[:, :, k], R', f_f[:, :, k])
        r_q[:, :, k] .-= u_q[:, :, k]
    end

    # solve for time derivative
    @timeit "trans. VDM" mul!(dudt[:, :, k], V', r_q[:, :, k])
    @timeit "mass solve" mass_matrix_solve!(solver.mass_solver,
        k,
        dudt[:, :, k],
        u_q[:, :, k])
    return dudt
end

@views @timeit "semi-disc. residual" function rhs_benchmark!(
        dudt::AbstractArray{Float64,
            3},
        u::AbstractArray{Float64, 3},
        solver::Solver{<:AbstractConservationLaw{d,
                FirstOrder},
            <:PhysicalOperators,
            <:AbstractMassMatrixSolver,
            <:StandardForm},
        t::Float64 = 0.0) where {d}
    @timeit "unpack" begin
        (; conservation_law, operators, connectivity, form) = solver
        (; inviscid_numerical_flux) = form
        (; f_q, f_f, u_q, u_f, temp, CI) = solver.preallocated_arrays
    end

    k = 1  # just one element

    @timeit "vandermonde" mul!(u_q[:, :, k], operators.V[k], u[:, :, k])
    @timeit "extrap solution" mul!(u_f[:, k, :], operators.R[k], u_q[:, :, k])

    @timeit "phys flux" physical_flux!(f_q[:, :, :, k], conservation_law, u_q[:, :, k])

    @timeit "num flux" numerical_flux!(f_f[:, :, k],
        conservation_law,
        inviscid_numerical_flux,
        u_f[:, k, :],
        u_f[CI[connectivity[:, k]], :],
        operators.n_f[k])

    @timeit "fill w zeros" fill!(view(dudt, :, :, k), 0.0)

    @inbounds for m in 1:d
        @timeit "volume operators" begin
            mul!(view(temp, :, :, k), operators.VOL[k][m], f_q[:, :, m, k])
            dudt[:, :, k] .+= temp[:, :, k]
        end
    end

    mul!(view(temp, :, :, k), operators.FAC[k], f_f[:, :, k])
    dudt[:, :, k] .+= temp[:, :, k]

    return dudt
end

function scaling_test_euler_2d(p::Int,
        M::Int,
        path = "./results/euler_benchmark/",
        parallelism = Threaded())
    path = new_path(path, true, true)

    mach_number = 0.4
    angle = 0.0
    L = 1.0
    γ = 1.4
    T = L / mach_number # end time
    strength = sqrt(2 / (γ - 1) * (1 - 0.75^(γ - 1)))

    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law,
        θ = angle,
        Ma = mach_number,
        β = strength,
        R = 1.0 / 10.0,
        x_0 = (L / 2, L / 2))

    form = FluxDifferencingForm(inviscid_numerical_flux = LaxFriedrichsNumericalFlux())

    println("building reference approximation...")
    reference_approximation = ReferenceApproximation(ModalTensor(p), Tri(),
        mapping_degree = p, N_plot = 25)

    println("generating mesh...")
    uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0, L), (0.0, L)),
        (M, M))

    println("warping mesh...")
    mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(1 / 16, (L, L)))

    println("building global discretization...")
    spatial_discretization = SpatialDiscretization(mesh, reference_approximation,
        project_jacobian = true)

    println("preprocessing...")
    ode = semidiscretize(conservation_law,
        spatial_discretization,
        exact_solution,
        form,
        (0.0, T),
        ReferenceOperator(),
        GenericTensorProductAlgorithm(),
        parallelism = parallelism)

    dudt = similar(ode.u0)
    println("solving...")
    b = @benchmark semi_discrete_residual!($dudt, $ode.u0, $ode.p, 0.0)
    min_time = minimum(b.times)
    med_time = median(b.times)
    println("min = ", min_time, " med = ", med_time)

    if !isfile(string(path, "minimum.jld2"))
        save_object(string(path, "minimum.jld2"), [min_time])
        save_object(string(path, "median.jld2"), [med_time])
    else
        minima = load_object(string(path, "minimum.jld2"))
        push!(minima, min_time)
        save_object(string(path, "minimum.jld2"), min_time)

        medians = load_object(string(path, "median.jld2"))
        push!(medians, med)
        save_object(string(path, "median.jld2"), med_time)
    end
end
