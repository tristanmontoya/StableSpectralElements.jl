abstract type AbstractQuadratureRule end
struct LGLQuadrature <: AbstractQuadratureRule end
struct LGQuadrature <: AbstractQuadratureRule end

function quadrature(::Line,
    quadrature_rule::LGQuadrature,
    num_quad_nodes::Int)
        return gauss_quad(0,0,num_quad_nodes-1) 
end

function quadrature(::Line, 
    ::LGLQuadrature,
    num_quad_nodes::Int)
        return gauss_lobatto_quad(0,0,num_quad_nodes-1)
end

function quadrature(::Quad,
    quadrature_rule::AbstractQuadratureRule,
    num_quad_nodes::Int)

    r1d, w1d = quadrature(Line(), 
        quadrature_rule, num_quad_nodes)
    mgw = meshgrid(w1d,w1d)
    mgr = meshgrid(r1d,r1d)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end