using StartUpDG
using CLOUD.ConservationLaws
using CLOUD.SpatialDiscretizations
using OrdinaryDiffEq

M = 5  # number of elements
a = 1.0  # advection velocity
p_map = 1  # degree of mapping
p = 2 # degree of discretization
n_quad_nodes = 4 # number of quadrature points

# define a constant-coefficient linear adection equation
conservation_law = ConstantLinearAdvectionEquation1D(a)

# define reference element for mesh
reference_element = RefElemData(
    Line(), 
    p_map, 
    quad_rule_vol=gauss_lobatto_quad(0, 0, n_quad_nodes-1))

# make a uniform periodic mesh on the domain Ω = (-1,1)
mesh = make_periodic(
    MeshData(uniform_mesh(Line(), M)..., reference_element))

# construct spatial discretization
spatial_discretization = SpatialDiscretization1D(
    conservation_law,
    mesh,
    reference_element,
    CollocatedLG(p),
    StrongConservationForm())

# make semi-discrete residual for use with SciML
#ode = semi_discrete_residual(spatial_discretization, 
#    t_initial, 
#    t_final)