abstract type AbstractQuadratureRule end
struct LGLQuadrature <: AbstractQuadratureRule
    q::Int
end
struct LGQuadrature <: AbstractQuadratureRule
    q::Int
end
struct LGRQuadrature <: AbstractQuadratureRule
    q::Int
end
struct JGLQuadrature <: AbstractQuadratureRule
    q::Int
end
struct JGQuadrature <: AbstractQuadratureRule
    q::Int
end

struct JGRQuadrature <: AbstractQuadratureRule
    q::Int
end

struct DefaultQuadrature <: AbstractQuadratureRule
    degree::Int
end

function meshgrid(x::Vector{Float64}, y::Vector{Float64})
    return ([x[j] for i in 1:length(y), j in 1:length(x)],
        [y[i] for i in 1:length(y), j in 1:length(x)])
end

function meshgrid(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64})
    return ([x[k] for i in 1:length(z), j in 1:length(y), k in 1:length(y)],
        [y[j] for i in 1:length(z), j in 1:length(y), k in 1:length(y)],
        [z[i] for i in 1:length(z), j in 1:length(y), k in 1:length(y)])
end


function quadrature(::Line, quadrature_rule::DefaultQuadrature)
    return quadrature(Line(), 
        LGQuadrature(ceil(Int, (quadrature_rule.degree-1)/2)))
end

function quadrature(::Tri, quadrature_rule::DefaultQuadrature)
    return quad_nodes_tri(quadrature_rule.degree)
end

function quadrature(::Tet, quadrature_rule::DefaultQuadrature)
    return quad_nodes_tet(quadrature_rule.degree)
end

function quadrature(::Line, quadrature_rule::LGQuadrature)
    return gauss_quad(0,0,quadrature_rule.q) 
end

function quadrature(::Line, quadrature_rule::LGLQuadrature)
    return gauss_lobatto_quad(0,0,quadrature_rule.q)
end

function quadrature(::Line, quadrature_rule::LGRQuadrature)
    z = zgrjm(quadrature_rule.q+1, 0.0, 0.0)
    return z, wgrjm(z, 0.0, 0.0)
end

function quadrature(::Line, quadrature_rule::JGQuadrature)
    z = zgj(quadrature_rule.q+1, 1.0, 0.0)
    return z, wgj(z, 1.0, 0.0) ./ (1 .- z)
end

function quadrature(::Line, quadrature_rule::JGRQuadrature)
    z = zgrjm(quadrature_rule.q+1, 1.0, 0.0)
    return z, wgrjm(z, 1.0, 0.0) ./ (1 .- z)
end

function quadrature(::Quad, quadrature_rule::AbstractQuadratureRule)
    r1d, w1d = quadrature(Line(), quadrature_rule)
    mgw = meshgrid(w1d,w1d)
    mgr = meshgrid(r1d,r1d)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::Quad,
    quadrature_rule::NTuple{2,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    mgw = meshgrid(w1d_1,w1d_2)
    mgr = meshgrid(r1d_1,r1d_2)
    w2d = @. mgw[1] * mgw[2] 
    return mgr[1][:], mgr[2][:], w2d[:]
end

function quadrature(::Hex,
    quadrature_rule::NTuple{3,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    r1d_3, w1d_3 = quadrature(Line(), quadrature_rule[3])
    mgw = meshgrid(w1d_1,w1d_2,w1d_3)
    mgr = meshgrid(r1d_1,r1d_2,r1d_3)
    w2d = @. mgw[1] * mgw[2] * mgw[3] 
    return mgr[1][:], mgr[2][:], mgr[3][:], w2d[:]
end

function quadrature(::Hex, quadrature_rule::AbstractQuadratureRule)
    return quadrature(Hex(), Tuple(quadrature_rule for m in 1:3))
end

function quadrature(::Tri, quadrature_rule::NTuple{2,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    mgw = meshgrid(w1d_1, w1d_2)
    mgr = meshgrid(r1d_1,r1d_2)
    w2d = @. mgw[1] * mgw[2] 
    return χ(Tri(), (mgr[1][:], mgr[2][:]))..., 
        (η -> 0.5*(1-η)).(mgr[2][:]) .* w2d[:]
end

function facet_node_ids(::Line, N::Int)
    return [1, N]
end

function facet_node_ids(::Quad, N::NTuple{2,Int})
    return [
        1:N[1];  # left
        (N[1]*(N[2]-1)+1):(N[1]*N[2]); # right 
        1:N[1]:(N[1]*(N[2]-1)+1);  # bottom
            N[1]:N[1]:(N[1]*N[2])]  # top
end

function facet_node_ids(::Hex, N::NTuple{3,Int})
    return [
        1:N[1];  # η1(-)
        (N[1]*(N[2]-1)+1):(N[1]*N[2]); # top 
        1:N[1]:(N[1]*(N[2]-1)+1);  # left
            N[1]:N[1]:(N[1]*N[2])]  # right
end