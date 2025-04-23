struct Plotter{d}
    x_plot::NTuple{d, Matrix{Float64}}
    V_plot::LinearMap
    N_e::Int
    directory_name::String
end

function Plotter(spatial_discretization::SpatialDiscretization{d},
        directory_name::String) where {d}
    path = new_path(directory_name, true, false)

    return Plotter{d}(spatial_discretization.x_plot,
        spatial_discretization.reference_approximation.V_plot,
        spatial_discretization.N_e,
        path)
end

@recipe function plot(spatial_discretization::SpatialDiscretization{1},
        sol::Array{Float64, 3};
        e = 1,
        exact_solution = nothing,
        time = 0.0)
    (; x_plot, N_e, reference_approximation) = spatial_discretization
    xlabel --> "\$x\$"
    label --> ["\$U^h(x,t)\$" "\$U(x,t)\$"]

    @series begin
        vec(vcat(x_plot[1], fill(NaN, 1, N_e))),
        vec(vcat(convert(Matrix, reference_approximation.V_plot * sol[:, e, :]),
            fill(NaN, 1, N_e)))
    end

    if !isnothing(exact_solution)
        @series begin
            vec(x_plot[1]), vec(evaluate(exact_solution, x_plot, time)[:, e, :])
        end
    end
end

@recipe function plot(spatial_discretization::SpatialDiscretization{1},
        sol::Vector{Array{Float64, 3}};
        e = 1,
        exact_solution = nothing,
        t = 0.0)
    (; x_plot, N_e, reference_approximation) = spatial_discretization
    xlabel --> "\$x\$"
    label --> ""

    for k in eachindex(sol)
        @series begin
            vec(vcat(x_plot[1], fill(NaN, 1, N_e))),
            vec(vcat(convert(Matrix, reference_approximation.V_plot * sol[k][:, e, :]),
                fill(NaN, 1, N_e)))
        end
    end

    if !isnothing(exact_solution)
        @series begin
            vec(x_plot[1]), vec(evaluate(exact_solution, x_plot, t)[:, e, :])
        end
    end
end
