abstract type AbstractQuadratureRule end
struct LGLQuadrature <: AbstractQuadratureRule end
struct LGQuadrature <: AbstractQuadratureRule end

function volume_quadrature(::Line,
    quadrature_rule::LGQuadrature;
    num_quad_nodes::Int)
        return gauss_quad(0,0,num_quad_nodes-1) 
end

function volume_quadrature(::Line, 
    ::LGLQuadrature,
    num_quad_nodes::Int)
        return gauss_lobatto_quad(0,0,num_quad_nodes-1)
end