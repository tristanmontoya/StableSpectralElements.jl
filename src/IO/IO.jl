module IO

    using LinearMaps: LinearMap
    using Plots: scatter
    using ..SpatialDiscretizations: SpatialDiscretization

    struct Plotter{d}
        x_plot::NTuple{d,Vector{Float64}}
        V_plot::LinearMap{Float64}
    end

    function Plotter(spatial_discretization::SpatialDiscretization{d}) where {d}
        
        map_to_plot = spatial_discretization.reference_element.Vp
        x_plot = Tuple(map_to_plot*spatial_discretization.mesh.x[m] 
            for m in 1:d)

        return Plotter{d}(x_plot, 
            spatial_discretization.reference_operators.V_plot)
    end
        
    function plot_function(f, plotter::Plotter{2})
        u = @. f(plotter.x_plot)

        scatter(plotter.x_plot[0],plotter.x_plot[1],
            u,zcolor=up,msw=0,leg=false,ratio=1,cam=(0,90))
    end

end