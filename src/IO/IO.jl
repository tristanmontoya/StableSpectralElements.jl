module IO

    using LinearMaps: LinearMap
    using Plots: plot, scatter, savefig
    using LaTeXStrings
    using OrdinaryDiffEq: ODEProblem, ODESolution

    using ..SpatialDiscretizations: SpatialDiscretization

    export Plotter, plot_function

    struct Plotter{d}
        x_plot::NTuple{d,Matrix{Float64}}
        V_plot::LinearMap
        directory_name::String
    end

    function Plotter(spatial_discretization::SpatialDiscretization{d},directory_name::String) where {d}
        mkpath(directory_name) 
        return Plotter{d}(spatial_discretization.x_plot, 
            spatial_discretization.reference_operators.V_plot, directory_name)
    end
    
    function Plotter(spatial_discretization::SpatialDiscretization{d}) where {d}
        mkpath("../plots") 
        return Plotter{d}(spatial_discretization.x_plot, 
            spatial_discretization.reference_operators.V_plot, "../plots/")
    end

    function plot_function(f::Function, plotter::Plotter{1}, file_name::String; label::String="U(x,t)", e::Int=1)
        p = plot(plotter.x_plot[1],f(plotter.x_plot)[e], leg=false, xlabel=latexstring("x"), ylabel=latexstring(label))
        savefig(p, string(plotter.directory_name, file_name))
    end

    function plot_function(f::Function, plotter::Plotter{2}, file_name::String;
        label::String="U(\\mathbf{x},t)", e::Int=1)
        scatter(plotter.x_plot[1],plotter.x_plot[2],
            f(plotter.x_plot),
            zcolor=up,msw=0,leg=false,ratio=1,cam=(0,90))
        savefig(p, string(plotter.directory_name, file_name))
    end

end