using StableSpectralElements, LinearAlgebra, Plots, SparseArrays, KrylovKit
using DelimitedFiles,Dates

function spectral_radius_tri(opertype::String,M_vec,p_vec)
    L=1.0
    d=2
    a = Tuple(1.0 for m in 1:d)

    conservation_law = LinearAdvectionEquation(a)

    mesh_perturb = 0.0/16
    p_cnt = 1;
    SR_vec = zeros(length(M_vec),length(p_vec));
    for p in p_vec
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
            elseif opertype == "ModalOmega"
                reference_approximation = ReferenceApproximation(
                    ModalMulti(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTensor"
                reference_approximation = ReferenceApproximation(
                    NodalTensor(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalOmega"
                reference_approximation = ReferenceApproximation(
                    NodalMulti(p), Tri(), mapping_degree=p)
            elseif opertype == "ModalDiagE"
                reference_approximation = ReferenceApproximation(
                    ModalMultiDiagE(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalDiagE"
                reference_approximation = ReferenceApproximation(
                    NodalMultiDiagE(p), Tri(), mapping_degree=p)
            elseif opertype == "MFNodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    MatrixFreeTPSSLGL(p), Tri(), mapping_degree=p)
            else
                error("opertype not supported")
        end

        M_cnt = 1
        for M in M_vec
            form = StandardForm(mapping_form=SkewSymmetricMapping(), inviscid_numerical_flux=LaxFriedrichsNumericalFlux(1.0))

            original_mesh = uniform_periodic_mesh(reference_approximation, Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d))

            mesh = warp_mesh(original_mesh, reference_approximation, ChanWarping(mesh_perturb,Tuple(L for m in 1:d)))

            spatial_discretization = SpatialDiscretization(mesh, reference_approximation, project_jacobian=true)

            strategy = ReferenceOperator()
            operator_algorithm = DefaultOperatorAlgorithm()

            solver = Solver(conservation_law,spatial_discretization, form,
                strategy, operator_algorithm, default_mass_matrix_solver(
                spatial_discretization, operator_algorithm), Serial());
            
            #solver_map = Matrix(LinearResidual(solver))
            #println("Computing eigenvalues for p = ", p)
            #F = eigen(solver_map)
            #SR_vec[cnt]=maximum(abs.(F.values))
            println("Computing eigenvalues for p = ", p)
            solver_map = LinearResidual(solver)
            f(x) = solver_map*x
#=             eigenvalues, _, info = eigsolve(f, size(solver_map,1), 1, :LM,
                tol=1e-2, verbosity=1, ishermitian=false,
                issymmetric=false) =#
                eigenvalues, _, info = eigsolve(f, size(solver_map,1), 1, :LM;
                tol=1e-2,
                maxiter=500,
                krylovdim=50,   # or even 100
                verbosity=1,
                ishermitian=false,
                issymmetric=false)        
            SR_vec[M_cnt,p_cnt] = maximum(abs.(eigenvalues))
            M_cnt += 1
        end

    p_cnt +=1
    
    end
    open("MatrixProperties/"*string(opertype)*"M"*join(M_vec, "")*"_p"*join(p_vec, "")*"SpectralRadius.txt", "w") do io
        println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "SR matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, SR_vec)
    end    
end

function spectral_radius_tet(opertype::String,M_vec,p_vec)
    L=1.0
    d=3
    a = Tuple(1.0 for m in 1:d)

    conservation_law = LinearAdvectionEquation(a)

    mesh_perturb = 0.0/16
    p_cnt = 1;
    SR_vec = zeros(length(M_vec),length(p_vec));
    for p in p_vec
        if opertype == "NodalTPSS"
            reference_approximation = ReferenceApproximation(
                NodalTPSS(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalTPSSOpt"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOpt(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSLGL(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalTPSSMinimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSMinimal(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalTPSSOptimal"
                reference_approximation = ReferenceApproximation(
                    NodalTPSSOptimal(p), Tet(), mapping_degree=p)
            elseif opertype == "ModalTensor"
                reference_approximation = ReferenceApproximation(
                    ModalTensor(p), Tet(), mapping_degree=p)
            elseif opertype == "ModalOmega"
                reference_approximation = ReferenceApproximation(
                    ModalMulti(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalTensor"
                reference_approximation = ReferenceApproximation(
                    NodalTensor(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalOmega"
                reference_approximation = ReferenceApproximation(
                    NodalMulti(p), Tet(), mapping_degree=p)
            elseif opertype == "ModalDiagE"
                reference_approximation = ReferenceApproximation(
                    ModalMultiDiagE(p), Tet(), mapping_degree=p)
            elseif opertype == "NodalDiagE"
                reference_approximation = ReferenceApproximation(
                    NodalMultiDiagE(p), Tet(), mapping_degree=p)
            elseif opertype == "MFNodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    MatrixFreeTPSSLGL(p), Tet(), mapping_degree=p)
            else
                error("opertype not supported")
        end
        M_cnt = 1
        for M in M_vec
            form = StandardForm(mapping_form=SkewSymmetricMapping(), inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0))

            original_mesh = uniform_periodic_mesh(reference_approximation, Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d))

            mesh = warp_mesh(original_mesh, reference_approximation, ChanWarping(mesh_perturb,Tuple(L for m in 1:d)))

            spatial_discretization = SpatialDiscretization(mesh, reference_approximation, project_jacobian=true)

            strategy = ReferenceOperator()
            operator_algorithm = DefaultOperatorAlgorithm()

            solver = Solver(conservation_law,spatial_discretization, form,
                strategy, operator_algorithm, default_mass_matrix_solver(
                spatial_discretization, operator_algorithm), Serial());
            
            #solver_map = Matrix(LinearResidual(solver))
            #println("Computing eigenvalues for p = ", p)
            #F = eigen(solver_map)
            #SR_vec[cnt]=maximum(abs.(F.values))
            
            println("Computing eigenvalues for p = ", p, " and M = ", M)
            solver_map = LinearResidual(solver)
            f(x) = solver_map*x
            eigenvalues, _, info = eigsolve(f, size(solver_map,1), 1, :LM;
            tol=1e-2,
            maxiter=500,
            krylovdim=50,   # or even 100
            verbosity=1,
            ishermitian=false,
            issymmetric=false)  
            
            SR_vec[M_cnt,p_cnt] = maximum(abs.(eigenvalues))

            M_cnt += 1
        end
    p_cnt +=1
    
end
    open("MatrixProperties/3D_"*string(opertype)*"M"*join(M_vec, "")*"_p"*join(p_vec, "")*"SpectralRadius.txt", "w") do io
    println(io, "Operator type: ", opertype)
        println(io, "p_vec: ", p_vec)
        println(io, "M_vec: ", M_vec)
        println(io, "SR matrix (rows=M_vec, cols=p_vec):")
        writedlm(io, SR_vec)
    end    
end

function sparsity(opertype::String,M,p_vec)
    L=1.0
    d=2
    a = Tuple(1.0 for m in 1:d)

    conservation_law = LinearAdvectionEquation(a)

    mesh_perturb = 1.0/16

    sparsity = zeros(length(p_vec));
    cnt = 1;
    for p in p_vec
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
            elseif opertype == "ModalOmega"
                reference_approximation = ReferenceApproximation(
                    ModalMulti(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalTensor"
                reference_approximation = ReferenceApproximation(
                    NodalTensor(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalOmega"
                reference_approximation = ReferenceApproximation(
                    NodalMulti(p), Tri(), mapping_degree=p)
            elseif opertype == "ModalDiagE"
                reference_approximation = ReferenceApproximation(
                    ModalMultiDiagE(p), Tri(), mapping_degree=p)
            elseif opertype == "NodalDiagE"
                reference_approximation = ReferenceApproximation(
                    NodalMultiDiagE(p), Tri(), mapping_degree=p)
            elseif opertype == "MFNodalTPSSLGL"
                reference_approximation = ReferenceApproximation(
                    MatrixFreeTPSSLGL(p), Tri(), mapping_degree=p)
            else
                error("opertype not supported")
        end

        form = StandardForm(mapping_form=SkewSymmetricMapping(), inviscid_numerical_flux=LaxFriedrichsNumericalFlux(0.0))

        original_mesh = uniform_periodic_mesh(reference_approximation, Tuple((0.0,L) for m in 1:d), Tuple(M for m in 1:d))

        mesh = warp_mesh(original_mesh, reference_approximation, ChanWarping(mesh_perturb,Tuple(L for m in 1:d)))

        spatial_discretization = SpatialDiscretization(mesh, reference_approximation, project_jacobian=true)

        strategy = ReferenceOperator()
        operator_algorithm = DefaultOperatorAlgorithm()

        solver = Solver(conservation_law,spatial_discretization, form,
            strategy, operator_algorithm, default_mass_matrix_solver(
            spatial_discretization, operator_algorithm), Serial());
        
        

        residual = LinearResidual(solver)
        A = sparse(Matrix(residual))
        println("nnz = ", nnz(A))
        
        sparsity[cnt] =nnz(A)/length(A);
        cnt +=1
        
    end

    return sparsity
end

spectral_radius_tri("ModalTensor",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("ModalDiagE",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("ModalOmega",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("NodalTensor",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("NodalDiagE",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("NodalOmega",[8,12,16,24],[1,2,3,4,5,6,7,8])
spectral_radius_tri("NodalTPSSLGL",[2,4,6,8,10,12],[1,2,3,4,5,6,7,8])
spectral_radius_tri("NodalTPSSOptimal",[1,2,3,4],[2,3,4,5,6])
spectral_radius_tri("NodalTPSSMinimal",[1,2,3,4],[2,3,4,5,6])

#spectral_radius_tet("ModalTensor",[1],[1,2,3,4,5])
#spectral_radius_tet("ModalDiagE",[1],[1,2,3,4,5])
#spectral_radius_tet("ModalOmega",[1],[1,2,3,4,5])
#spectral_radius_tet("NodalTensor",[1],[1,2,3,4,5])
#spectral_radius_tet("NodalDiagE",[1],[1,2,3,4,5])
#spectral_radius_tet("NodalOmega",[1],[1,2,3,4,5])
#spectral_radius_tet("NodalTPSSLGL",[1],[1,2,3,4,5])
#spectral_radius_tet("NodalTPSSOptimal",[1],[3,4,5,6])
#spectral_radius_tet("NodalTPSSMinimal",[1],[3,4,5,6])
