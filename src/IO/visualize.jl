struct Plotter{d}
    x_plot::NTuple{d,Matrix{Float64}}
    V_plot::LinearMap
    N_el::Int
    directory_name::String
end

function Plotter(spatial_discretization::SpatialDiscretization{d},directory_name::String) where {d}
    path = new_path(directory_name, true, false)

    return Plotter{d}(spatial_discretization.x_plot, 
        spatial_discretization.reference_approximation.V_plot, spatial_discretization.N_el, path)
end

function visualize(sol::Array{Float64,3}, 
    plotter::Plotter{1}, file_name::String; e::Int=1,
    exact_solution::Union{AbstractInitialData{1},Nothing}=nothing,
    label::String="U^h(x,t)", label_exact::String="U(x,t)")

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    u = convert(Matrix, V_plot * sol[:,e,:])
    if !isnothing(exact_solution)
        p = plot(vec(x_plot[1]), vec(evaluate(exact_solution,x_plot)[:,e,:]), 
            label=latexstring(label_exact))
        plot!(p, vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
            vec(vcat(u,fill(NaN,1,N_el))), 
            label=latexstring(label), xlabel=latexstring("x"))
    else 
        p = plot(vec(vcat(x_plot[1],fill(NaN,1,N_el))), 
        vec(vcat(u,fill(NaN,1,N_el))), 
        label=latexstring(label), xlabel=latexstring("x"))
    end

    savefig(p, string(directory_name, file_name))   
    return p
end

function visualize(sol::Array{Float64,3}, plotter::Plotter{2},
    file_name::String; label::String="U(\\mathbf{x},t)", e::Int=1)

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    u = convert(Matrix, V_plot * sol[:,e,:])

    p = scatter(x_plot[1],x_plot[2], u,
        zcolor=u,msw=0,leg=false,ratio=1,cam=(0,90))
    savefig(p, string(directory_name, file_name))
    return p
end

function visualize(solution::AbstractInitialData, plotter::Plotter{2},
    file_name::String; label::String="U(\\mathbf{x},t)", e::Int=1)

    @unpack x_plot, V_plot, N_el, directory_name = plotter

    u = evaluate(solution, x_plot)[:,e,:]

    p = scatter(x_plot[1],x_plot[2], u,
        zcolor=u,msw=0,leg=false,ratio=1,cam=(0,90))
    savefig(p, string(directory_name, file_name))
    return p
end