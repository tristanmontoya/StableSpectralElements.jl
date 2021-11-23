module IO

    using LinearMaps: LinearMap
    using Plots: plot, plot!, scatter, savefig
    using LaTeXStrings
    using OrdinaryDiffEq: ODEProblem, ODESolution
    using ..SpatialDiscretizations: SpatialDiscretization, apply_to_all_dof

    export Plotter, visualize

    struct Plotter{d}
        x_plot::NTuple{d,Matrix{Float64}}
        V_plot::LinearMap
        N_el::Int
        directory_name::String
    end

    function Plotter(spatial_discretization::SpatialDiscretization{d},directory_name::String) where {d}
        mkpath(directory_name) 
        return Plotter{d}(spatial_discretization.x_plot, 
            spatial_discretization.reference_operators.V_plot, spatial_discretization.N_el, directory_name)
    end
    
    function Plotter(spatial_discretization::SpatialDiscretization{d}) where {d}
        mkpath("../plots") 
        return Plotter{d}(spatial_discretization.x_plot, 
            spatial_discretization.reference_operators.V_plot, 
            spatial_discretization.N_el, "../plots/")
    end

    function visualize(f::Function, plotter::Plotter{1}, file_name::String; label::String="U(x,t)", e::Int=1)
        p = plot(vec(plotter.x_plot[1]),
            vec(f(plotter.x_plot)[e]),xlabel=latexstring("x"), label=latexstring(label))
        savefig(p, string(plotter.directory_name, file_name))
        return p
    end

    function visualize(f::Function, plotter::Plotter{2}, file_name::String;
        label::String="U(\\mathbf{x},t)", e::Int=1)
        scatter(plotter.x_plot[1],plotter.x_plot[2],
            f(plotter.x_plot)[e],
            zcolor=up,msw=0,leg=false,ratio=1,cam=(0,90))
        savefig(p, string(plotter.directory_name, file_name))
        return p
    end

    function visualize(sol::Array{Float64,3}, 
        plotter::Plotter{1}, file_name::String; e::Int=1,
        exact_solution::Union{Function,Nothing}=nothing,
        label::String="U^h(x,t)", label_exact::String="U(x,t)")

        u = vec(apply_to_all_dof(
            fill(plotter.V_plot, plotter.N_el), sol)[:,e,:])
        
        p = plot(vec(plotter.x_plot[1]),u, label=latexstring(label), 
            xlabel=latexstring("x"))

        if !isnothing(exact_solution)
            plot!(p, vec(plotter.x_plot[1]),
                vec(exact_solution(plotter.x_plot)[e]), 
                label=latexstring(label_exact))
        end
        
        savefig(p, string(plotter.directory_name, 
        file_name))   
        return p

    end
end