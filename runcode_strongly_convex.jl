# Strong-convexity experiment for the JMLR revision (K = 10 replication).
#
# Addresses Referee 1 weakness #6 and Referee 3 comment #4.
# Modifies the LS objective with a Tikhonov term (μ/2)||x_i||^2 so each f_i is
# μ-strongly convex, then runs RandomEdge with a sweep of constant step sizes α.
# For each α we run K independent RandomEdge trajectories (graph and data
# fixed; only the algorithm randomness varies) and plot the mean trajectory of
# ||x^(t) - x*||^2 with a thin band of all K replicates. Theorem 5.13 predicts
# linear convergence to a plateau whose height scales linearly with α.

include("./src.jl")
using Convex, SCS, LaTeXStrings

Random.seed!(42)

# ---- problem instance -----------------------------------------------------
num_group         = 3
num_nodes_group   = [10, 10, 10]
num_nodes         = sum(num_nodes_group)
dim               = 20
num_training_node = 15 * ones(Int64, num_nodes)
num_testing_node  = 10 * ones(Int64, num_nodes)
λ                 = 1.0
μ                 = 0.5

# strongly-convex loss & gradient: add (μ/2)||x||^2 to LS
loss_LS_sc(X, Y, x) = loss_LS(X, Y, x) + (μ / 2) * dot(x, x)
grad_LS_sc(X, Y, x) = grad_LS(X, Y, x) .+ μ .* x
loss_LS_sc_cvx(X::Matrix, Y::Vector, x) = loss_LS_cvx(X, Y, x) + (μ / 2) * sumsquares(x)

# ---- generate a connected graph (fixed across all α and replicates) -------
local g, x_true, trainX, trainY, degrees
while true
    global g, trainX, trainY, degrees
    g, _, trainX, trainY, _, _, _, degrees, _ = gen_graph(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node,
        gen_truex, gen_data_LS; noise=1e-2, same_prob=0.5, diff_prob=0.05)
    is_connected(g) && break
end
println("Graph: n=$(nv(g)), m=$(ne(g))")

# ---- centralized strongly-convex solution via Convex.jl --------------------
d_per_node = size(trainX[1], 2)
x_var = Convex.Variable(d_per_node, num_nodes)
loss_expr = sum(loss_LS_sc_cvx(trainX[i], trainY[i], x_var[:, i]) for i = 1:num_nodes)
reg_expr  = sum(reg2_cvx(x_var[:, i], x_var[:, j]) for i = 1:num_nodes for j in neighbors(g, i) if i < j)
problem   = minimize(loss_expr + λ * reg_expr)
solve!(problem, () -> SCS.Optimizer(); silent=true)
x_star = evaluate(x_var)
println("Centralized optimum: $(problem.optval)")

# ---- RandomEdge: K trajectories per step size -----------------------------
αs        = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
maxiter   = 30000
K         = 10
num_edges = ne(g)
neigh_    = [neighbors(g, i) for i = 1:num_nodes]

# results[α] :: Matrix{Float64}, size maxiter × K
results = Dict{Float64, Matrix{Float64}}()

for (kα, α) in enumerate(αs)
    R = zeros(maxiter, K)
    for k = 1:K
        Random.seed!(100 * kα + k)
        x_iter = initial_x(d_per_node, num_nodes)
        newx   = zeros(d_per_node, num_nodes)
        for t = 1:maxiter
            Threads.@threads for i = 1:num_nodes
                g_i        = grad_LS_sc(trainX[i], trainY[i], x_iter[:, i])
                newx[:, i] = x_iter[:, i] .- α .* g_i
            end
            Threads.@threads for i = 1:num_nodes
                if degrees[i] == 0 || rand() >= degrees[i] / num_edges
                    x_iter[:, i] = newx[:, i]
                else
                    j           = rand(neigh_[i], 1)[1]
                    x_iter[:, i] = pair_prox2(newx[:, i], newx[:, j], α * num_edges, λ)
                end
            end
            R[t, k] = sum((x_iter .- x_star).^2)
        end
        @printf "α=%.0e k=%2d: final ‖x-x*‖²=%+10.4e\n" α k R[end, k]
    end
    results[α] = R
end

# ---- plot: mean trajectory + thin band of K replicates --------------------
colors = [:black, :red, :blue, :green, :purple]
p = plot(yscale=:log10, xlabel="Iteration", ylabel=L"\|x^{(t)} - x^*\|^2",
        legend=:topright, dpi=300, tickfontsize=12, guidefontsize=14,
        legendfontsize=11, size=(700, 420))
for (kα, α) in enumerate(αs)
    R = results[α]
    # thin transparent lines for each replicate
    for k = 1:K
        plot!(p, 1:maxiter, R[:, k]; color=colors[kα], linewidth=0.4, alpha=0.15,
              label="")
    end
    # thick mean line
    plot!(p, 1:maxiter, vec(mean(R, dims=2)); label=latexstring("\\alpha=$(α)"),
          color=colors[kα], linewidth=1.8)
end
title!(p, "Strong convexity: linear convergence to a neighborhood (K=$K)")
savefig(p, "fig_strongly_convex.png")
println("Saved fig_strongly_convex.png")

# ---- log the plateau heights for the response letter ---------------------
plateau_window = 1000
plateau = Dict{Float64, Tuple{Float64, Float64}}()
for α in αs
    final_vals = vec(mean(results[α][end-plateau_window+1:end, :], dims=1))
    plateau[α] = (mean(final_vals), std(final_vals))
end
println("\nPlateau height (‖x-x*‖² averaged over last $plateau_window iters):")
for α in αs
    m, s = plateau[α]
    @printf "  α=%.0e -> mean=%.4e ± %.4e (mean/α = %.4e)\n" α m s (m / α)
end
