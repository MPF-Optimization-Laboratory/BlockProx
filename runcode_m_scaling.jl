# m-scaling experiment for the JMLR revision (K = 10 replication).
#
# Addresses Referee 3 comment #4.
# Fixes n AND the underlying data; varies only the Erdős–Rényi edge probability
# so m = |E| varies. For each edge probability p we generate K independent
# connected ER graphs on the fixed node set, solve the centralized optimum
# per graph, and run RandomEdge with a fixed constant step size α to its steady
# state. The plateau ‖x_T - x*‖^2 (averaged over the last `burn_in` iterates) is
# aggregated as mean ± log-symmetric error bars across the K replicates.
# Theorem 5.13 predicts this residual scales as O(m^2 c^2 α / μ).

include("./src.jl")
using Convex, SCS, LaTeXStrings

# ---- fixed problem parameters --------------------------------------------
num_nodes     = 25
dim           = 15
λ             = 1.0
α             = 3e-3
maxiter       = 20000
burn_in       = 2000
noise         = 1e-2
K             = 10                  # replications per edge probability
samples_per_n = 15

# ---- fixed data (independent of graph topology) --------------------------
Random.seed!(2026)
x_true_full = randn(dim + 1)
trainX = Vector{Matrix{Float64}}(undef, num_nodes)
trainY = Vector{Vector{Float64}}(undef, num_nodes)
for i = 1:num_nodes
    trainX[i], trainY[i] = gen_data_LS(samples_per_n, dim, x_true_full, noise)
end
d_per_node = size(trainX[1], 2)   # dim + 1

# ---- edge probability sweep ----------------------------------------------
edge_probs    = [0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95]
m_per_p       = [Float64[] for _ in edge_probs]
plateau_per_p = [Float64[] for _ in edge_probs]

for (idx, p) in enumerate(edge_probs)
    for k = 1:K
        # generate a connected ER graph (vary seed across (p, k))
        Random.seed!(3000 + 100 * idx + k)
        local g, degrees
        while true
            g = Graphs.erdos_renyi(num_nodes, p)
            is_connected(g) && break
        end
        degrees = [length(neighbors(g, i)) for i = 1:num_nodes]
        m_val   = ne(g)

        # centralized optimum: depends on the graph (regularizer term)
        x_var     = Convex.Variable(d_per_node, num_nodes)
        loss_expr = sum(loss_LS_cvx(trainX[i], trainY[i], x_var[:, i]) for i = 1:num_nodes)
        reg_expr  = sum(reg2_cvx(x_var[:, i], x_var[:, j])
                        for i = 1:num_nodes for j in neighbors(g, i) if i < j)
        problem   = minimize(loss_expr + λ * reg_expr)
        solve!(problem, () -> SCS.Optimizer(); silent=true)
        x_star    = evaluate(x_var)

        # RandomEdge with constant step size; track ‖x_T - x*‖²
        Random.seed!(7000 + 100 * idx + k)
        x_iter   = initial_x(d_per_node, num_nodes)
        newx     = zeros(d_per_node, num_nodes)
        neigh_   = [neighbors(g, i) for i = 1:num_nodes]
        residual = zeros(maxiter)
        for t = 1:maxiter
            Threads.@threads for i = 1:num_nodes
                g_i        = grad_LS(trainX[i], trainY[i], x_iter[:, i])
                newx[:, i] = x_iter[:, i] .- α .* g_i
            end
            Threads.@threads for i = 1:num_nodes
                if degrees[i] == 0 || rand() >= degrees[i] / m_val
                    x_iter[:, i] = newx[:, i]
                else
                    j            = rand(neigh_[i], 1)[1]
                    x_iter[:, i] = pair_prox2(newx[:, i], newx[:, j], α * m_val, λ)
                end
            end
            residual[t] = sum((x_iter .- x_star).^2)
        end
        plateau = mean(residual[end-burn_in+1:end])

        push!(m_per_p[idx], m_val)
        push!(plateau_per_p[idx], plateau)
        @printf "p=%.2f k=%2d -> m=%4d  plateau=%.4e\n" p k m_val plateau
    end
end

# ---- aggregate -----------------------------------------------------------
mean_m       = [mean(v) for v in m_per_p]
std_m        = [std(v)  for v in m_per_p]
mean_plat    = [mean(v) for v in plateau_per_p]
std_log_plat = [std(log.(v)) for v in plateau_per_p]
yerr_low     = mean_plat .* (1 .- exp.(-std_log_plat))
yerr_high    = mean_plat .* (exp.(std_log_plat) .- 1)

println("\nAggregate (mean ± 1 std on log-plateau across K=$K):")
for (idx, p) in enumerate(edge_probs)
    @printf "  p=%.2f: mean m=%6.1f (±%.1f), mean plateau=%.4e (log-std=%.3f)\n" p mean_m[idx] std_m[idx] mean_plat[idx] std_log_plat[idx]
end

# ---- log-log plot --------------------------------------------------------
# Faint scatter of all K·|p| individual realizations
all_m_flat       = reduce(vcat, m_per_p)
all_plateau_flat = reduce(vcat, plateau_per_p)
p_plot = scatter(all_m_flat, all_plateau_flat;
                 xscale=:log10, yscale=:log10,
                 xlabel=L"m = |\mathcal{E}|",
                 ylabel=L"\|x_T - x^*\|^2\;(\mathrm{steady\ state})",
                 markersize=3, color=:gray, alpha=0.4, markerstrokewidth=0,
                 label="individual realizations",
                 dpi=300, tickfontsize=12, guidefontsize=14, legendfontsize=11,
                 size=(700, 420))

# Mean per p with log-symmetric vertical error bars
scatter!(p_plot, mean_m, mean_plat;
         yerror=(yerr_low, yerr_high),
         markersize=6, color=:black, markerstrokewidth=1.2,
         label="mean ± 1 std (K=$K)")

# reference slope-2 line anchored at the smallest-m mean
m_ref       = collect(range(minimum(mean_m), maximum(mean_m); length=20))
plateau_ref = mean_plat[1] * (m_ref ./ mean_m[1]).^2
plot!(p_plot, m_ref, plateau_ref; linestyle=:dash, color=:red, linewidth=2,
      label=L"slope\ 2\ (\mathrm{theory})")
title!(p_plot, "m-scaling: steady-state residual vs |E|")
savefig(p_plot, "fig_m_scaling.png")
println("Saved fig_m_scaling.png")

# linear fit on log-log (using means)
logm    = log.(mean_m)
logplat = log.(mean_plat)
A       = hcat(ones(length(logm)), logm)
β       = A \ logplat
@printf "Log-log fit on means: slope = %.3f, intercept = %.3f\n" β[2] β[1]
