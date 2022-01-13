abstract type AbstractQuadratureRule{d} end

struct LGLQuadrature <: AbstractQuadratureRule{1} end
struct LGQuadrature <: AbstractQuadratureRule{1} end

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
    quadrature_rule::AbstractQuadratureRule{1},
    num_quad_nodes::Int)

    r1d, w1d = quadrature(Line(), 
        quadrature_rule, num_quad_nodes)
    mgw = meshgrid(w1d,w1d)
    mgr = meshgrid(r1d,r1d)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::Quad,
    quadrature_rule_1::AbstractQuadratureRule{1},
    quadrature_rule_2::AbstractQuadratureRule{1},
    num_quad_nodes_1::Int,
    num_quad_nodes_2::Int)

    r1d_1, w1d_1 = quadrature(Line(), 
        quadrature_rule_1, num_quad_nodes_1)
    r1d_2, w1d_2 = quadrature(Line(), 
        quadrature_rule_2, num_quad_nodes_2)
    mgw = meshgrid(w1d_1,w1d_2)
    mgr = meshgrid(r1d_1,r1d_2)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::DuffyTri,
    quadrature_rule_1::AbstractQuadratureRule{1},
    quadrature_rule_2::AbstractQuadratureRule{1},
    num_quad_nodes_1::Int,
    num_quad_nodes_2::Int)

    J =  0.5*(1 .- r1d_2)
    r1d_1, w1d_1 = quadrature(Line(), 
        quadrature_rule_1, num_quad_nodes_1)
    r1d_2, w1d_2 = quadrature(Line(), 
        quadrature_rule_2, num_quad_nodes_2)
    mgw = meshgrid(w1d_1, J.* w1d_2)
    mgr = meshgrid(χ(DuffyTri(),(r1d_1, r1d_2))...)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end