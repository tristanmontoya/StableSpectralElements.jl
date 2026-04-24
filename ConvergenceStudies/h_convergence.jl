using StableSpectralElements, OrdinaryDiffEq, .Threads
using Plots, Plots.PlotMeasures, TimerOutputs

function lin_advec_h_converge(p::Int, opertype::String, M_vec, plt)
    a = (1.0,1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1.0  # end time
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    # set mesh refinement levels 
    err_vec = zeros(length(M_vec),1)

    # set CFL number 
    CFL = 0.1

    # accumulator 
    cnt = 1

    # loop over meshes
    for M in M_vec

        #define reference approximation
        if opertype == "NodalTPSS"
            reference_approximation = ReferenceApproximation(
                NodalTPSS(p), Tri(), mapping_degree=p)
        elseif opertype == "NodalTPSSOpt"
            reference_approximation = ReferenceApproximation(
                NodalTPSSOpt(p), Tri(), mapping_degree=p)
        elseif opertype == "NodalTPSSLGL"
            reference_approximation = ReferenceApproximation(
                NodalTPSSLGL(p), Tri(), mapping_degree=p)
        else
            error("opertype should be 'NodalTPSS', 'NodalTPSSOpt', or 'NodalTPSSLGL'")
        end

        # define discretization form, upwind flux 
        form = StandardForm(mapping_form=SkewSymmetricMapping(), 
            inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))

        # define mesh and warp
        uniform_mesh = uniform_periodic_mesh(reference_approximation,
            ((0.0,L),(0.0,L)), (M,M))
        mesh = warp_mesh(uniform_mesh, reference_approximation, 0.0, L)

        # define spatial discretization
        spatial_discretization = SpatialDiscretization(mesh, 
            reference_approximation, project_jacobian=true)

        # results path
        results_path = save_project(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T),
            "results/advection_2d/", overwrite=true, clear=true);

        # discretize to system of ODEs
        ode_problem = semidiscretize(conservation_law,
            spatial_discretization, exact_solution, form, (0.0, T))
        
        # time step definition
        h = L/sqrt(reference_approximation.N_p * spatial_discretization.N_e)
        dt = CFL * h / sqrt(a[1]^2 + a[2]^2)
        
        # solve 
        sol = solve(ode_problem, DP5(), 
            adaptive=false, dt=dt, save_everystep=false, 
            callback=save_callback(results_path, (0.0,T), floor(Int, T/(dt*50))))

        # calculate and record error
        error_analysis = ErrorAnalysis(results_path, conservation_law, 
            spatial_discretization)
        err_vec[cnt] = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])

        # update accumulator 
        cnt +=1
    end

    # calculate slopes 
    for i = 1:length(M_vec)-1
        println("Slope")
        println((log(err_vec[i])-log(err_vec[i+1]))/(log(M_vec[i+1])-log(M_vec[i])))
    end

    # plot solution if needed
    if plt == true    
        plt = plot(M_vec,err_vec, xscale = :log10, yscale = :log10, lw = 2, marker = :o, label = "Error vs h")
        xlabel!("h")
        ylabel!("L2 Error")
        display(plt)
    end

    return err_vec

end

function lin_advec_h_converge_allp(opertype::String, p_vec, M_vec)
    a = (1.0,1.0)  # advection velocity
    L = 1.0  # domain length
    T = 1.0  # end time
    conservation_law = LinearAdvectionEquation(a)
    exact_solution = InitialDataSine(1.0,(2π/L, 2π/L));

    # set mesh refinement levels 
    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))
    # set CFL number 
    CFL = 0.1

    # accumulator for p 
    pcnt = 1

    # loop over p
    for p in p_vec

        # accumulator for M 
        Mcnt = 1

        # loop over meshes
        for M in M_vec

            #define reference approximation
            if opertype == "NodalTPSS"
                reference_approximation = ReferenceApproximation(
                    NodalTPSS(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSOpt"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOpt(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSLGL(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSMinimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSMinimal(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSOptimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOptimal(p), Tri(), mapping_degree=p)
            elseif opertype == "ModalTensor"
                reference_approximation = ReferenceApproximation(
                    ModalTensor(p), Tri(), mapping_degree=p)
            else
                error("opertype should be 'NodalTPSS', 'NodalTPSSOpt', or 'NodalTPSSLGL'")
            end

            # define discretization form, upwind flux 
            form = StandardForm(mapping_form=SkewSymmetricMapping(), 
                inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))

            # define mesh and warp
            uniform_mesh = uniform_periodic_mesh(reference_approximation,
                ((0.0,L),(0.0,L)), (M,M))
            mesh = warp_mesh(uniform_mesh, reference_approximation, 0.1, L)

            # define spatial discretization
            spatial_discretization = SpatialDiscretization(mesh, 
                reference_approximation, project_jacobian=true)
            
            # results path
            results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/advection_2d/", overwrite=true, clear=true);

            # discretize to system of ODEs
            ode_problem = semidiscretize(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T))
            dof_vec[Mcnt,pcnt]= size(ode_problem.u0)[1]*size(ode_problem.u0)[3]
            # time step definition
            h = L/sqrt(reference_approximation.N_p * spatial_discretization.N_e)
            dt = CFL * h / sqrt(a[1]^2 + a[2]^2)
            
            # solve 
            sol = solve(ode_problem, DP5(), 
                adaptive=false, dt=dt, save_everystep=false, 
                callback=save_callback(results_path, (0.0,T), floor(Int, T/(dt*50))))

            # calculate and record error
            error_analysis = ErrorAnalysis(results_path, conservation_law, 
                spatial_discretization) 
            err_vec[Mcnt,pcnt] = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1])

            # update accumulator 
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    return err_vec, dof_vec

end

function euler_h_converge_allp(opertype::String, p_vec, M_vec)
    mach_number = 0.8
    angle = 0.0
    L = 1.0
    γ=1.4
    T = L/mach_number # end time
    strength = sqrt(2/(γ-1)*(1-0.75^(γ-1))) # for central value of ρ=0.75
    
    conservation_law = EulerEquations{2}(γ)
    exact_solution = IsentropicVortex(conservation_law, θ=angle,
        Ma=mach_number, β=strength, R=1.0/10.0, x_0=(L/2,L/2));

    # set mesh refinement levels 
    err_vec = zeros(length(M_vec),length(p_vec))
    dof_vec = zeros(length(M_vec),length(p_vec))

    # accumulator for p 
    pcnt = 1

    # loop over p
    for p in p_vec

        # accumulator for M 
        Mcnt = 1

        # loop over meshes
        for M in M_vec

            #define reference approximation
            if opertype == "NodalTPSS"
                reference_approximation = ReferenceApproximation(
                    NodalTPSS(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSOpt"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOpt(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSLGL(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSMinimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSMinimal(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTPSSOptimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOptimal(p), Tri(), mapping_degree=p)
            elseif opertype == "ModalTensor"
                reference_approximation = ReferenceApproximation(
                    ModalTensor(p), Tri(), mapping_degree=p)
            else
                error("opertype should be 'NodalTPSS', 'NodalTPSSOpt', or 'NodalTPSSLGL'")
            end

            #form = FluxDifferencingForm(inviscid_numerical_flux=LaxFriedrichsNumericalFlux())
            form = StandardForm(mapping_form=SkewSymmetricMapping(), inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))
            
            reference_approximation = ReferenceApproximation(NodalTPSSMinimal(p), Tri(), mapping_degree=p)
            
            uniform_mesh = uniform_periodic_mesh(reference_approximation, ((0.0,L),(0.0,L)), (M,M))
            
            mesh = warp_mesh(uniform_mesh, reference_approximation, ChanWarping(1.0/16, (L,L)))
            
            spatial_discretization = SpatialDiscretization(mesh, reference_approximation)
            
            results_path = save_project(conservation_law,
                spatial_discretization, exact_solution, form, (0.0, T),
                "results/euler_vortex_2d/", overwrite=true, clear=true)
            
            ode = semidiscretize(conservation_law, spatial_discretization, 
                exact_solution, form, (0.0, T), parallelism=Serial());
            
            dof_vec[Mcnt,pcnt]= size(ode.u0)[1]*size(ode.u0)[3]
            C_t = 0.01
            dt = C_t * (L/M) / (mach_number*p^2)
            sol = solve(ode, DP8(), adaptive=false, dt=dt, save_everystep=false, 
                callback=save_callback(results_path, (0.0,T), 5))
            
            error_analysis = ErrorAnalysis(results_path, conservation_law, 
                spatial_discretization, DefaultQuadrature(35))
            
            err_vec[Mcnt,pcnt] = Float64(analyze(error_analysis, last(sol.u), exact_solution, T)[1][1])

            # update accumulator 
            Mcnt +=1
        end

        # update accumulator
        pcnt += 1
    end

    return err_vec, dof_vec

end
err, dof = euler_h_converge_allp("NodalTPSSMinimal",[2,3,4],[2,3,4])

display(err)
display(dof)