Base.@kwdef struct WeakConservationForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
    mapping_form::MappingForm = SkewSymmetricMapping()
    inviscid_numerical_flux::AbstractInviscidNumericalFlux =
        LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = NoTwoPointFlux()
end

Base.@kwdef struct SplitConservationForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
    mapping_form::MappingForm = SkewSymmetricMapping()
    inviscid_numerical_flux::AbstractInviscidNumericalFlux =
        LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = NoTwoPointFlux()
end

Base.@kwdef struct StrongConservationForm{MappingForm,TwoPointFlux} <: AbstractResidualForm{MappingForm,TwoPointFlux}
    mapping_form::MappingForm = SkewSymmetricMapping()
    inviscid_numerical_flux::AbstractInviscidNumericalFlux =
        LaxFriedrichsNumericalFlux()
    viscous_numerical_flux::AbstractViscousNumericalFlux = BR1()
    two_point_flux::TwoPointFlux = NoTwoPointFlux()
end

"""
Make operators for weak conservation form
"""
function make_operators(spatial_discretization::SpatialDiscretization{1}, 
    ::WeakConservationForm{StandardMapping,<:AbstractTwoPointFlux},
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver())

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack D, V, Vf, R, W, B, N_p, N_q, N_f = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, nJf = spatial_discretization.geometric_factors
    op(A::LinearMap) = make_operator(A, operator_algorithm)

    operators = Array{DiscretizationOperators}(undef, N_e)
    @inbounds for k in 1:N_e
        VOL = (op(D[1]' * W),)
        FAC = op(-R' * B)
        SRC = Diagonal(W * J_q[:,k])
        NTR = (Diagonal(nJf[1][:,k]) * op(R),)

        operators[k] = DiscretizationOperators{1}(VOL, FAC, SRC, NTR,
            mass_matrix(V,W,Diagonal(J_q[:,k]), mass_matrix_solver),
            op(V), op(R), (nJf[1][:,k],), N_p, N_q, N_f)
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::WeakConservationForm{StandardMapping,<:AbstractTwoPointFlux},
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver()) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D, N_p, N_q, N_f = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, J_f, nJf = spatial_discretization.geometric_factors
    op(A::LinearMap) = make_operator(A, operator_algorithm)

    operators = Array{DiscretizationOperators}(undef, N_e)
    @inbounds for k in 1:N_e
        VOL = Tuple(sum(op(D[m]') * Diagonal(W * Λ_q[:,m,n,k]) for m in 1:d)    
            for n in 1:d)
        FAC = op(-R') * Diagonal(B * J_f[:,k])
        SRC = Diagonal(W * J_q[:,k])

        n_f = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        NTR = Tuple(LinearMap(zeros(N_f,N_q)) for m in 1:d)

        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, NTR,
            mass_matrix(V,W,Diagonal(J_q[:,k]), mass_matrix_solver), 
            op(V), op(R), n_f, N_p, N_q, N_f)
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::WeakConservationForm{SkewSymmetricMapping,<:AbstractTwoPointFlux},
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver()) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D, N_p, N_q, N_f = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, J_f, nJf = spatial_discretization.geometric_factors
    op(A::LinearMap) = make_operator(A, operator_algorithm)

    operators = Array{DiscretizationOperators}(undef, N_e)
    @inbounds for k in 1:N_e
        VOL = Tuple(sum(
            op(D[m]') * Diagonal(0.5 * W * Λ_q[:,m,n,k]) -
                        Diagonal(0.5 * W * Λ_q[:,m,n,k]) * op(D[m])  # skew part
                    for m in 1:d) +
                op(R') * Diagonal(0.5 * B * nJf[n][:,k]) * op(R)  # sym part
                    for n in 1:d)
        FAC = op(-R') * Diagonal(B * J_f[:,k])
        SRC = Diagonal(W * J_q[:,k])

        n_f = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        NTR = Tuple(ZeroMap(N_f,N_q) for m in 1:d)

        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, NTR,
            mass_matrix(V,W,Diagonal(J_q[:,k]), mass_matrix_solver), 
            op(V), op(R), n_f, N_p, N_q, N_f)
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::StrongConservationForm{SkewSymmetricMapping,<:AbstractTwoPointFlux},
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver()) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D, N_p, N_q, N_f = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, J_f, nJf = spatial_discretization.geometric_factors
    op(A::LinearMap) = make_operator(A, operator_algorithm)

    operators = Array{DiscretizationOperators}(undef, N_e)
    @inbounds for k in 1:N_e
        VOL = Tuple(-sum(
            Diagonal(0.5* W * Λ_q[:,m,n,k]) * op(D[m]) -
                op(D[m]') * Diagonal(0.5 * W * Λ_q[:,m,n,k]) # skew part
                    for m in 1:d) -
                op(R') * Diagonal(0.5*B * nJf[n][:,k]) * op(R)  # sym part
                    for n in 1:d)
        FAC = op(-R') * Diagonal(B * J_f[:,k])
        SRC = Diagonal(W * J_q[:,k])
        n_f = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        NTR = Tuple(Diagonal(-n_f[m]) * op(R) for m in 1:d)

        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, NTR,
            mass_matrix(V,W,Diagonal(J_q[:,k]), mass_matrix_solver), 
            op(V), op(R), n_f, N_p, N_q, N_f)
    end
    return operators
end

function make_operators(spatial_discretization::SpatialDiscretization{d}, 
    ::SplitConservationForm{SkewSymmetricMapping,<:AbstractTwoPointFlux},
    operator_algorithm::AbstractOperatorAlgorithm=DefaultOperatorAlgorithm(),
    mass_matrix_solver::AbstractMassMatrixSolver=CholeskySolver()) where {d}

    @unpack N_e, M, reference_approximation = spatial_discretization
    @unpack V, Vf, R, W, B, D, N_p, N_q, N_f = reference_approximation
    @unpack nrstJ = reference_approximation.reference_element
    @unpack J_q, Λ_q, J_f, nJf = spatial_discretization.geometric_factors
    op(A::LinearMap) = make_operator(A, operator_algorithm)

    operators = Array{DiscretizationOperators}(undef, N_e)
    @inbounds for k in 1:N_e
        VOL = Tuple(sum(
            op(D[m]') * Diagonal(0.5 * W * Λ_q[:,m,n,k]) -
                        Diagonal(0.5 * W * Λ_q[:,m,n,k]) * op(D[m]) 
                    for m in 1:d) for n in 1:d)
        FAC = op(-R') * Diagonal(B * J_f[:,k])
        SRC = Diagonal(W * J_q[:,k])
        n_f = Tuple(nJf[m][:,k] ./ J_f[:,k] for m in 1:d)
        NTR = Tuple(Diagonal(-0.5*n_f[m]) * op(R) for m in 1:d)

        operators[k] = DiscretizationOperators{d}(VOL, FAC, SRC, NTR,
            mass_matrix(V,W,Diagonal(J_q[:,k]), mass_matrix_solver), 
            op(V), op(R), n_f, N_p, N_q, N_f)
    end
    return operators
end