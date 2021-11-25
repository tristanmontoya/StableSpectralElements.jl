struct StrongConservationForm <: AbstractResidualForm end

function  element_residual(
    conservation_law::ConservationLaw{d, N_eq},
    spatial_discretization::SpatialDiscretization{d},
    form::StrongConservationForm,
    operator_storage::PhysicalOperatorStorage) where {d, N_eq}

    # give reference operators short names for clarity
    D = spatial_discretization.reference_operators.D_strong
    P = spatial_discretization.reference_operators.P
    V = spatial_discretization.reference_operators.V
    R = spatial_discretization.reference_operators.R
    L = spatial_discretization.reference_operators.L


    # store physical volume and facet operators
    volume_ops = Array{LinearMap}(undef, d, spatial_discretization.N_el)
    facet_ops = Array{LinearMap}(undef, spatial_discretization.N_el)
    for k in 1:spatial_discretization.N_el
        J = LinearMap(diagm(spatial_discretization.mesh.J[:,k]))
        if d == 1
            volume_ops[1,k] = inv(P*J*V)*D[1]*P
        end
        facet_ops[k] =  inv(P*J*V)*L
    end

    function res(u,k)
        # See Andreas Klockner thesis.
        
    end

    return u -> res(u,k)
end