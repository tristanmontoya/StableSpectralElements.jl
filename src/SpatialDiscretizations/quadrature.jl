abstract type AbstractQuadratureRule{d} end
struct LGLQuadrature <: AbstractQuadratureRule{1} end
struct LGQuadrature <: AbstractQuadratureRule{1} end
struct JGRQuadrature <: AbstractQuadratureRule{1}
    a::Float64
    b::Float64
end

function quadrature(::Line, ::LGQuadrature, N::Int)
    return gauss_quad(0,0,N-1) 
end

function quadrature(::Line, ::LGLQuadrature, N::Int)
    return gauss_lobatto_quad(0,0,N-1)
end

function quadrature(::Line, quadrature_rule::JGRQuadrature, N::Int)
    z = zgrjm(N, quadrature_rule.a, quadrature_rule.b)
    return z, wgrjm(z, quadrature_rule.a, quadrature_rule.b)
end

function quadrature(::Quad,
    quadrature_rule::AbstractQuadratureRule{1},
    N::Int)

    r1d, w1d = quadrature(Line(), quadrature_rule, N)
    mgw = meshgrid(w1d,w1d)
    mgr = meshgrid(r1d,r1d)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::Quad,
    quadrature_rule::NTuple{2,AbstractQuadratureRule{1}}, N::NTuple{2,Int})

    r1d_1, w1d_1 = quadrature(Line(), 
        quadrature_rule[1], N[1])
    r1d_2, w1d_2 = quadrature(Line(), 
        quadrature_rule[2], N[2])
    mgw = meshgrid(w1d_1,w1d_2)
    mgr = meshgrid(r1d_1,r1d_2)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::CollapsedTri,
    quadrature_rule::NTuple{2,AbstractQuadratureRule{1}}, N::NTuple{2,Int})

    r1d_1, w1d_1 = quadrature(Line(), 
        quadrature_rule[1], N[1])
    r1d_2, w1d_2 = quadrature(Line(), 
        quadrature_rule[2], N[2])
    mgw = meshgrid(w1d_1, w1d_2)
    mgr = meshgrid(r1d_1,r1d_2)
    w2d = @. mgw[1] * mgw[2] 
    return χ(CollapsedTri(), (mgr[1][:], mgr[2][:]))..., 
        (η -> 0.5*(1-η)).(mgr[2][:]) .* w2d[:]
end

function facet_node_ids(::Line, N::Int)
    return [1, N]
end

function facet_node_ids(::Quad, N::NTuple{2,Int})
    return [1:N[1]:(N[1]*(N[2]-1)+1);  # left
            N[1]:N[1]:(N[1]*N[2]);  # right
            1:N[1];  # bottom
            (N[1]*(N[2]-1)+1):(N[1]*N[2])]  # top 
end