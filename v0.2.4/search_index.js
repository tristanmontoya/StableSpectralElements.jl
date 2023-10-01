var documenterSearchIndex = {"docs":
[{"location":"SpatialDiscretizations/#Module-SpatialDiscretizations","page":"SpatialDiscretizations","title":"Module SpatialDiscretizations","text":"","category":"section"},{"location":"SpatialDiscretizations/#Overview","page":"SpatialDiscretizations","title":"Overview","text":"","category":"section"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"Discretizations in StableSpectralElements.jl are constructed by first building a local approximation on a canonical reference element, denoted generically as hatOmega subset mathbbR^d, and using a bijective transformation bmX^(kappa)  hatOmega rightarrow Omega^(kappa) to construct the approximation on each physical element of the mesh mathcalT^h =  Omega^(kappa)_kappa in 1N_e in terms of the associated operators on the reference element. An example of such a mapping is shown below.","category":"page"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"(Image: Mesh mapping)","category":"page"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"In order to define the different geometric reference elements, existing subtypes of AbstractElemShape from StartUpDG.jl (e.g. Line, Quad, Hex, Tri, and Tet) are used and re-exported by StableSpectralElements.jl. For example, we have ","category":"page"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"beginaligned\nhatOmega_mathrmline = -11\nhatOmega_mathrmquad = -11^2\nhatOmega_mathrmhex  = -11^3 \nhatOmega_mathrmtri = big bmxi in -11^2  xi_1 + xi_2 leq 0 big\nhatOmega_mathrmtet = big bmxi in -11^3  xi_1 + xi_2 + xi_3 leq -1 big\nendaligned","category":"page"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"These element types are used in the constructor for StableSpectralElements.jl's ReferenceApproximation type, along with a subtype of AbstractApproximationType specifying the nature of the local approximation (and, optionally, the associated volume and facet quadrature rules).","category":"page"},{"location":"SpatialDiscretizations/","page":"SpatialDiscretizations","title":"SpatialDiscretizations","text":"All the information used to define the spatial discretization on the physical domain Omega is contained within a SpatialDiscretization structure, which is constructed using a ReferenceApproximation and a MeshData from StartUpDG.jl, which are stored as the fields reference_approximation and mesh. When the constructor for a SpatialDiscretization is called, the grid metrics are computed and stored in a GeometricFactors structure, with the corresponding field being geometric_factors. ","category":"page"},{"location":"ConservationLaws/#Module-ConservationLaws","page":"ConservationLaws","title":"Module ConservationLaws","text":"","category":"section"},{"location":"ConservationLaws/#Overview","page":"ConservationLaws","title":"Overview","text":"","category":"section"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"The equations to be solved are defined by subtypes of AbstractConservationLaw on which functions such as physical_flux and numerical_flux are dispatched. Objects of type AbstractConservationLaw contain two type parameters, d and PDEType, the former denoting the spatial dimension of the problem, which is inherited by all subtypes, and the latter being a subtype of AbstractPDEType denoting the particular type of PDE being solved, which is either FirstOrder or SecondOrder. Whereas first-order problems remove the dependence of the flux tensor on the solution gradient in order to obtain systems of the form","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"partial_t underlineU(bmxt) + bmnabla_bmx cdot underlinebmF(underlineU(bmxt)) = underline0","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"second-order problems are treated by StableSpectralElements.jl as first-order systems of the form ","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"beginaligned\nunderlinebmQ(bmxt) - bmnabla_bmx underlineU(bmxt) = underline0\npartial_t underlineU(bmxt) + bmnabla_bmx cdot underlinebmF(underlineU(bmxt) underlinebmQ(bmxt)) = underline0\nendaligned","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"Currently, the linear advection and advection-diffusion equations, the inviscid and viscous Burgers' equations, and the compressible Euler equations are supported by StableSpectralElements.jl, but any system of the above form can in principle be implemented, provided that appropriate physical and numerical fluxes are defined.","category":"page"},{"location":"ConservationLaws/#Equations","page":"ConservationLaws","title":"Equations","text":"","category":"section"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"Listed below are partial differential equations supported by StableSpectralElements.jl.","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"CurrentModule = ConservationLaws","category":"page"},{"location":"ConservationLaws/","page":"ConservationLaws","title":"ConservationLaws","text":"    LinearAdvectionEquation\n    LinearAdvectionDiffusionEquation\n    InviscidBurgersEquation\n    ViscousBurgersEquation\n    EulerEquations","category":"page"},{"location":"ConservationLaws/#StableSpectralElements.ConservationLaws.LinearAdvectionEquation","page":"ConservationLaws","title":"StableSpectralElements.ConservationLaws.LinearAdvectionEquation","text":"LinearAdvectionEquation(a::NTuple{d,Float64}) where {d}\n\nDefine a linear advection equation of the form\n\npartial_t U(bmxt) + bmnabla cdot big( bma U(bmxt) big) = 0\n\nwith a constant advection velocity bma in R^d. A specialized constructor LinearAdvectionEquation(a::Float64) is provided for the one-dimensional case.\n\n\n\n\n\n","category":"type"},{"location":"ConservationLaws/#StableSpectralElements.ConservationLaws.LinearAdvectionDiffusionEquation","page":"ConservationLaws","title":"StableSpectralElements.ConservationLaws.LinearAdvectionDiffusionEquation","text":"LinearAdvectionDiffusionEquation(a::NTuple{d,Float64}, b::Float64) where {d}\n\nDefine a linear advection-diffusion equation of the form\n\npartial_t U(bmxt) + bmnabla cdot big( bma U(bmxt) - b bmnabla U(bmxt)big) = 0\n\nwith a constant advection velocity bma in R^d and diffusion coefficient b in R^+. A specialized constructor LinearAdvectionDiffusionEquation(a::Float64, b::Float64) is provided for the one-dimensional case.\n\n\n\n\n\n","category":"type"},{"location":"ConservationLaws/#StableSpectralElements.ConservationLaws.InviscidBurgersEquation","page":"ConservationLaws","title":"StableSpectralElements.ConservationLaws.InviscidBurgersEquation","text":"InviscidBurgersEquation(a::NTuple{d,Float64}) where {d}\n\nDefine an inviscid Burgers' equation of the form\n\npartial_t U(bmxt) + bmnabla cdot big(tfrac12bma U(bmxt)^2 big) = 0\n\nwhere bma in R^d. A specialized constructor InviscidBurgersEquation() is provided for the one-dimensional case with a = (1.0,).\n\n\n\n\n\n","category":"type"},{"location":"ConservationLaws/#StableSpectralElements.ConservationLaws.ViscousBurgersEquation","page":"ConservationLaws","title":"StableSpectralElements.ConservationLaws.ViscousBurgersEquation","text":"x     ViscousBurgersEquation(a::NTuple{d,Float64}, b::Float64) where {d}\n\nDefine a viscous Burgers' equation of the form\n\npartial_t U(bmxt) + bmnabla cdot big(tfrac12bma U(bmxt)^2 - b bmnabla U(bmxt)big) = 0\n\nwhere bma in R^d and b in R^+. A specialized constructor ViscousBurgersEquation(b::Float64) is provided for the one-dimensional case with a = (1.0,).\n\n\n\n\n\n","category":"type"},{"location":"ConservationLaws/#StableSpectralElements.ConservationLaws.EulerEquations","page":"ConservationLaws","title":"StableSpectralElements.ConservationLaws.EulerEquations","text":"EulerEquations{d}(γ::Float64) where {d}\n\nDefine an Euler system governing compressible, adiabatic fluid flow, taking the form\n\nfracpartialpartial tleftbeginarrayc\nrho(bmx t) \nrho(bmx t) V_1(bmx t) \nvdots \nrho(bmx t) V_d(bmx t) \nE(bmx t)\nendarrayright+sum_m=1^d fracpartialpartial x_mleftbeginarrayc\nrho(bmx t) V_m(bmx t) \nrho(bmx t) V_1(bmx t) V_m(bmx t)+P(bmx t) delta_1 m \nvdots \nrho(bmx t) V_d(bmx t) V_m(bmx t)+P(bmx t) delta_d m \nV_m(bmx t)(E(bmx t)+P(bmx t))\nendarrayright=underline0\n\nwhere rho(bmxt) in mathbbR is the fluid density, bmV(bmxt) in mathbbR^d is the flow velocity, E(bmxt) in mathbbR is the total energy per unit volume, and the pressure is given for an ideal gas with constant specific heat as\n\nP(bmxt) = (gamma - 1)Big(E(bmxt) - frac12rho(bmxt) lVert bmV(bmxt)rVert^2Big)\n\nThe specific heat ratio is specified as a parameter γ::Float64, which must be greater than unity.\n\n\n\n\n\n","category":"type"},{"location":"#StableSpectralElements.jl","page":"Home","title":"StableSpectralElements.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StableSpectralElements.jl, formerly known as CLOUD.jl (Conservation Laws on Unstructured Domains), is a Julia framework for the numerical solution of partial differential equations of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"partial_t underlineU(bmxt) + bmnabla_bmx cdot underlinebmF(underlineU(bmxt) bmnabla_bmxunderlineU(bmxt)) = underline0","category":"page"},{"location":"","page":"Home","title":"Home","text":"for t in (0T) with T in mathbbR^+  and bmx in Omega subset mathbbR^d, subject to appropriate initial and boundary conditions, where underlineU(bmxt) is the vector of solution variables and underlinebmF(underlineU(bmxt)bmnabla_bmxunderlineU(bmxt)) is the flux tensor containing advective and/or diffusive contributions.  These equations are spatially discretized on curvilinear unstructured grids using energy-stable and entropy-stable discontinuous spectral element methods in order to generate ODEProblem objects suitable for time integration using OrdinaryDiffEq.jl within the SciML ecosystem. StableSpectralElements.jl also includes postprocessing tools employing WriteVTK.jl for generating .vtu files, allowing for visualization of high-order numerical solutions on unstructured grids using ParaView or other software. Shared-memory parallelization is supported through multithreading.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The functionality provided by StartUpDG.jl for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, StableSpectralElements.jl implements dispatched strategies for semi-discrete operator evaluation using LinearMaps.jl, allowing for the efficient matrix-free application of tensor-product operators.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Discretizations satisfying the summation-by-parts property employing nodal as well as modal bases are implemented, with the latter allowing for efficient and low-storage inversion of the dense elemental mass matrices arising from curvilinear meshes through the use of weight-adjusted approximations. ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StableSpectralElements.jl is a registered Julia package (compatible with Julia versions 1.8 and higher) and can be installed by entering the following commands within the REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg; Pkg.add(\"StableSpectralElements\")","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We recommend that users refer to the following Jupyter notebooks (included in the examples directory) for examples of how to use StableSpectralElements.jl:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Linear advection-diffusion equation in 1D\nInviscid Burgers' equation in 1D with energy-conservative scheme\nEuler equations in 1D with entropy-conservative Gauss collocation methods\nLinear advection equation in 2D\nIsentropic Euler vortex in 2D with entropy-stable modal scheme on triangles\nLinear advection equation in 3D","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This software is released under the GPLv3 license.","category":"page"}]
}