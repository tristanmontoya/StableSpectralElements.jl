# `CLOUD.jl` - Conservation Laws on Unstructured Domains

`CLOUD.jl` is a Julia framework for high-order discretizations of conservation laws of the form

 `∂ₜu + ∇⋅(F¹(u) + F²(u,∇u)) = s`

## Usage
A Jupyter notebook containing example solutions of the 2D linear advection equation on curvilinear triangular and quadrilateral meshes using modal DG and nodal DGSEM schemes, respectively, is provided in `examples/advection_2D.ipynb`.

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
