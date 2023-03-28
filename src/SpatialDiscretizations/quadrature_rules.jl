abstract type AbstractQuadratureRule end

struct GaussLobattoQuadrature <: AbstractQuadratureRule
    q::Int
    a::Int
    b::Int
end

struct GaussQuadrature <: AbstractQuadratureRule
    q::Int
    a::Int
    b::Int
end

struct GaussRadauQuadrature <: AbstractQuadratureRule
    q::Int
    a::Int
    b::Int
end

struct DefaultQuadrature <: AbstractQuadratureRule
    degree::Int
end

@inline LGLQuadrature(q::Int) = GaussLobattoQuadrature(q,0,0)
@inline LGQuadrature(q::Int) = GaussQuadrature(q,0,0)
@inline LGRQuadrature(q::Int) = GaussRadauQuadrature(q,0,0)

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

function quadrature(::Line, quadrature_rule::GaussLobattoQuadrature)
    @unpack q,a,b = quadrature_rule
    z = zglj(q+1, Float64(a), Float64(b))
    return z, wglj(z, Float64(a), Float64(b))
end

function quadrature(::Line, quadrature_rule::GaussQuadrature)
    @unpack q,a,b = quadrature_rule
    z = zgj(q+1, Float64(a), Float64(b))
    return z, wgj(z, Float64(a), Float64(b))
end

function quadrature(::Line, quadrature_rule::GaussRadauQuadrature)
    @unpack q,a,b = quadrature_rule
    z = zgrjm(q+1, Float64(a), Float64(b))
    return z, wgrjm(z, Float64(a), Float64(b))
end
function quadrature(::Quad, quadrature_rule::AbstractQuadratureRule)
    r1d, w1d = quadrature(Line(), quadrature_rule)
    w_grid = meshgrid(w1d,w1d)
    r_grid = meshgrid(r1d,r1d)
    w2d = w_grid[1] .* w_grid[2] 
    return r_grid[1][:], r_grid[2][:], w2d[:]
end

function quadrature(::Quad,
    quadrature_rule::NTuple{2,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    w_grid = meshgrid(w1d_1,w1d_2)
    r_grid = meshgrid(r1d_1,r1d_2)
    w2d = w_grid[1] .* w_grid[2] 
    return r_grid[1][:], r_grid[2][:], w2d[:]
end

function quadrature(::Hex,
    quadrature_rule::NTuple{3,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    r1d_3, w1d_3 = quadrature(Line(), quadrature_rule[3])
    w_grid = meshgrid(w1d_1,w1d_2,w1d_3)
    r_grid = meshgrid(r1d_1,r1d_2,r1d_3)
    w2d = w_grid[1] .* w_grid[2] .* w_grid[3] 
    return r_grid[1][:], r_grid[2][:], r_grid[3][:], w2d[:]
end

function quadrature(::Hex, quadrature_rule::AbstractQuadratureRule)
    return quadrature(Hex(), Tuple(quadrature_rule for m in 1:3))
end

function quadrature(::Tri, quadrature_rule::NTuple{2,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    w_grid = meshgrid(w1d_1, w1d_2)
    r_grid = meshgrid(r1d_1,r1d_2)
    w2d = w_grid[1] .* w_grid[2] 
    if ((quadrature_rule[1].a, quadrature_rule[1].b) == (0,0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (0,0))  
        return χ(Tri(), (r_grid[1][:], r_grid[2][:]))..., 
            0.5*(η -> (1-η)).(r_grid[2][:]) .* w2d[:]
    elseif ((quadrature_rule[1].a, quadrature_rule[1].b) == (0,0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (1,0))
        return χ(Tri(), (r_grid[1][:], r_grid[2][:]))..., 0.5*w2d[:]
    else 
        @error "Chosen Jacobi weight not supported" 
    end
end

function quadrature(::Tet, quadrature_rule::NTuple{3,AbstractQuadratureRule})
    r1d_1, w1d_1 = quadrature(Line(), quadrature_rule[1])
    r1d_2, w1d_2 = quadrature(Line(), quadrature_rule[2])
    r1d_3, w1d_3 = quadrature(Line(), quadrature_rule[3])
    w_grid = meshgrid(w1d_1,w1d_2,w1d_3)
    r_grid = meshgrid(r1d_1,r1d_2,r1d_3)
    w2d = w_grid[1] .* w_grid[2] .* w_grid[3] 

    if ((quadrature_rule[1].a, quadrature_rule[1].b) == (0,0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (0,0) &&
        (quadrature_rule[3].a, quadrature_rule[3].b) == (0,0))
        return χ(Tet(), (r_grid[1][:], r_grid[2][:], r_grid[3][:]))...,
        0.25(η -> (1-η)).(r_grid[2][:]) .* (η -> ((1-η))^2).(r_grid[3][:]) .* w2d[:]
    elseif ((quadrature_rule[1].a, quadrature_rule[1].b) == (0,0) &&
        (quadrature_rule[2].a, quadrature_rule[2].b) == (0,0) &&
        (quadrature_rule[3].a, quadrature_rule[3].b) == (1,0))
        return χ(Tet(), (r_grid[1][:], r_grid[2][:], r_grid[3][:]))...,
        0.25(η -> (1-η)).(r_grid[2][:]) .* (η -> (1-η)).(r_grid[3][:]) .* w2d[:]
    else 
        @error "Chosen Jacobi weight not supported" 
    end
end