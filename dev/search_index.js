var documenterSearchIndex = {"docs":
[{"location":"#CLOUD.jl:-Conservation-Laws-on-Unstructured-Domains","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"CLOUD.jl is a Julia framework for the numerical solution of partial differential equations of the form","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"fracpartial underlineU(bmxt)partial t + bmnabla_bmx cdot underlinebmF(underlineU(bmxt) bmnabla_bmxunderlineU(bmxt)) = underline0","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"subject to appropriate initial and boundary conditions, where underlineU(bmxt) is the vector of solution variables and underlinebmF(underlineU(bmxt)bmnabla_bmxunderlineU(bmxt)) is the flux tensor containing advective and/or diffusive contributions.  These equations are spatially discretized on curvilinear unstructured grids using discontinuous spectral element methods with the summation-by-parts property in order to generate ODEProblem objects suitable for time integration using OrdinaryDiffEq.jl within the SciML ecosystem. ","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"The functionality provided by StartUpDG.jl for the handling of mesh data structures, polynomial basis functions, and quadrature nodes is employed throughout this package. Moreover, CLOUD.jl employs dynamically dispatched strategies for semi-discrete operator evaluation using LinearMaps.jl, allowing for the efficient matrix-free application of tensor-product operators, including those associated with collapsed-coordinate formulations on triangles.","category":"page"},{"location":"#Installation","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"Installation","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"To install CLOUD.jl, open a julia session and enter:","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"julia> import Pkg\n\njulia> Pkg.add(\"https://github.com/tristanmontoya/CLOUD.jl.git\")","category":"page"},{"location":"#Basic-Usage","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"Basic Usage","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"As this documentation is currently a work in progress, we recommend that users refer to the following Jupyter notebooks for examples of how to use CLOUD.jl:","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"Linear advection-diffusion equation in 1D\nLinear advection equation in 2D","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"More detailed tutorials will be added soon!","category":"page"},{"location":"#Conservation-Laws","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"Conservation Laws","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"Wherever possible, CLOUD.jl separates the physical problem definition from the numerical discretization. The equations to be solved are defined by subtypes of AbstractConservationLaw on which functions such as physical_flux and numerical_flux are dispatched. Objects of type AbstractConservationLaw contain two type parameters, d and PDEType, the former denoting the spatial dimension of the problem, which is inherited by all subtypes, and the latter being a subtype of AbstractPDEType denoting the particular type of PDE being solved, which is either FirstOrder or SecondOrder. Second-order problems are treated by CLOUD.jl as first-order systems of the form ","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"beginaligned\nunderlinebmQ(bmxt) - bmnabla_bmx underlineU(bmxt) = underline0\nfracpartial underlineU(bmxt)partial t + bmnabla_bmx cdot underlinebmF(underlineU(bmxt) underlinebmQ(bmxt)) = underline0\nendaligned","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"CLOUD.jl also supports source terms of the form underlineS(bmxt), specifically for code verification using the method of manufactured solutions.","category":"page"},{"location":"#Approximation-on-the-Reference-Element","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"Approximation on the Reference Element","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"Discretizations in CLOUD.jl are constructed by first building a local approximation on a canonical reference element, denoted generically as hatOmega subset mathbbR^d, and using a bijective transformation bmX^(kappa)  hatOmega rightarrow Omega^(kappa) to construct the approximation on each physical element of the mesh mathcalT^h =  Omega^(kappa)_kappa=1^N_e in terms of the associated operators on the reference element. In order to define the different geometric reference elements, existing subtypes of AbstractElemShape from StartUpDG.jl (e.g. Line, Quad, and Tri) are used and re-exported by CLOUD.jl. For example, we have ","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"beginaligned\nhatOmega_mathrmline = -11\nhatOmega_mathrmquad = -11^2\nhatOmega_mathrmtri = big bmxi in -11^2  xi_1 + xi_2 leq 1 big\nendaligned","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"These element types are used in the constructor for CLOUD.jl's ReferenceApproximation type, along with a subtype of AbstractApproximationType specifying the nature of the local approximation (and, optionally, the associated volume and facet quadrature rules). As an example, we can construct a collapsed-edge tensor-product spectral-element method of degree 4 on the reference triangle by first loading the CLOUD.jl package and then using the appropriate constructor:","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"julia> using CLOUD\n\njulia> ref_tri = ReferenceApproximation(CollapsedSEM(4), Tri())","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"Using CLOUD.jl's built-in plotting recipes, we can easily visualize such a discretization:","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"julia> using Plots\n\njulia> plot(ref_elem_tri, grid_connect=true)","category":"page"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"(Image: CollapsedSEM)","category":"page"},{"location":"#License","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"License","text":"","category":"section"},{"location":"","page":"CLOUD.jl: Conservation Laws on Unstructured Domains","title":"CLOUD.jl: Conservation Laws on Unstructured Domains","text":"This software is released under the GPLv3 license.","category":"page"}]
}
