using CSV, DataFrames
using Distances: Euclidean, pairwise, Haversine

# include("./macros.jl")
include("./src.jl")

function housing_graph_generator(; num_nodes_test::Integer=0)
    # The file path of the data.
    path = "./Sacramentorealestatetransactions_Normalized.csv"
    # Read the data.
    df = CSV.read(path, DataFrame)
    # The number of all nodes in the dataset.
    num_nodes = size(df, 1)
    # The number of training nodes, i.e., the number of nodes in the graph.
    num_nodes_train = num_nodes - num_nodes_test

    # The sorted indices of data for training.
    # The reason for sorting this set is that we need it to be sorted to use
    # `deleteat!`.
    inds_train = randperm(num_nodes)[1:num_nodes_train] |> sort
    # The indices of data for testing.
    # inds_test = setdiff(1:num_nodes, inds_train)
    # Normally, we use the above code, but for efficiency, we use the following
    # code, which is faster according to tests.
    # Note that the following code requires inds_train to be sorted.
    inds_test = deleteat!(collect(1:num_nodes), inds_train)

    # The training data.
    df_train = df[inds_train, :]
    # The testing data.
    df_test = df[inds_test, :]

    # The locations for each node in the training graph.
    locations = df_train[!, [:latitude, :longitude]] |> Matrix
    # The distance matrix based on Haversine distance.
    dists_ = pairwise(Haversine(3961), locations')

    # The graph.
    g = SimpleGraph(num_nodes_train)
    # The training dataset.
    trainX = [zeros(1, 4) for i = 1:num_nodes_train]
    # The training output.
    trainY = df_train[!, :price_normalized] |> Vector

    for i = 1:num_nodes_train
        trainX[i][1:3] = df_train[i, [:beds, :baths, :sq__ft]] |> Vector
        trainX[i][4] = 1
        # Add 5 neighbors for each node.
        for each in partialsortperm(dists_[i, :], 2:6)
            add_edge!(g, i, each)
        end
    end
    # The degrees.
    # We note that the degree for each node is not simply 5, although every node
    # is connected to the closest 5 nodes, there are some nodes are connected
    # more than 5 nodes.
    degrees = [degree(g, i) for i = 1:num_nodes_train]


    # The weights, 5 / dist according to the paper, but it was a lie because the
    # code of the paper actually uses 1 / dist.
    wts = 1 ./ (dists_ .+ 1)

    return g, degrees, trainX, trainY, wts, df_test
end

function grad_housing(X::Matrix, Y::Real, x::Vector; μ::Real=0.5)
    first_term = 2 * (X * x .- Y)[1] * X |> vec
    second_term = 2 * μ * x
    second_term[end] = 0
    return first_term + second_term
end

function loss_housing(X::Matrix, Y::Real, x::Vector; μ::Real=0.5)
    return (X * x .- Y)[1]^2 + μ * norm(x[1:end-1], 2)^2
end

function loss_housing_cvx(X::Matrix, Y::Real, x::T; μ::Real=0.5) where T <: Convex.AbstractExpr
    return sumsquares(X * x - Y) + μ * sumsquares(x[1:end-1])
end

function ss_ook2(iter::Integer; init::Real=1e-3)
    return init / iter
end

function ss_ooqk2(iter::Integer; init::Real=1e-2)
    return init / sqrt(iter+1)
end

# ss_housing(iter::Integer) = ss_gap(iter; gap=150, init=1e-3, desc=1.1)
ss_housing(iter::Integer) = ss_ooqk2(iter; init=3e-3)

ss_const_housing(iter::Integer) = 1e-2
ss_const_housing_dgd(iter::Integer) = 1e-2

g, degrees, trainX, trainY, wts, df_test = housing_graph_generator()

λ = 1.0
maxiter = 25000
disp_freq = 2 * maxiter # Avoid displaying
num_nodes = nv(g)
dim = size(trainX[1], 2)

function loss_housing_dgd(
    X::Matrix{<:Real},
    Y::Real,
    x::Vector{<:Real},
    i::Int,
    g::SimpleGraph,
    λ::Real,
    reg::Function;
    μ::Real=0.5,
)
    # Dimension.
    dim = size(X, 2)
    # Local loss function value.
    val = (X * x[(i - 1) * dim + 1 : i * dim] .- Y)[1]^2 + 
                μ * norm(x[(i - 1) * dim + 1 : i * dim - 1], 2)^2

    # Regularizer.
    reg_ = 0
    for j in neighbors(g, i)
        reg_ += reg(x[(i - 1) * dim + 1 : i * dim], 
                    x[(j - 1) * dim + 1 : j * dim])
    end

    # Notice that each regularizer counts twice: one in the src, the other in
    # the dst, so we use λ / 2.
    return val + λ / 2 * reg_
end

function grad_housing_dgd(
    X::Matrix{<:Real},
    Y::Real,
    x::Vector{<:Real},
    i::Int,
    g::SimpleGraph,
    λ::Real,
    reg::Function;
    μ::Real=0.5,
)
    ∇ = gradient(x_ -> loss_housing_dgd(X, Y, x_, i, g, λ, reg; μ=μ), x)[1]
    return ∇ === nothing ? zero(x) : ∇
end

function loss_housing_walkman(
    X::Matrix{<:Real},
    Y::Real,
    x::Vector{<:Real},
    i::Int;
    μ::Real=0.5,
)
    # Dimension.
    dim = size(X, 2)
    # Local loss function value.
    val = (X * x[(i - 1) * dim + 1 : i * dim] .- Y)[1]^2 + 
                μ * norm(x[(i - 1) * dim + 1 : i * dim - 1], 2)^2
    
    return val
end

function grad_housing_walkman(
    X::Matrix{<:Real},
    Y::Real,
    x::Vector{<:Real},
    i::Int;
    μ::Real=0.5,
)
    ∇ = gradient(x_ -> loss_housing_walkman(X, Y, x_, i; μ), x)[1]
    return ∇ === nothing ? zero(x) : ∇
end

# Convex.jl
x_cvx, optval = whole_cvx(g, trainX, trainY, λ, loss_housing_cvx, reg2_cvx; wt=wts)
@printf "Convex.jl: %+10.7e\n" optval

# Block Proximal
# disp_freq = 1
x_blockprox, obj_blockprox, comms_blockprox = blockprox(g, trainX, trainY, λ,
    degrees, initial_x, ss_housing, grad_housing, pair_prox2, loss_housing, reg2;
    maxiter=maxiter, disp_freq=disp_freq, wt=wts)
@printf "BlockProx: %+10.7e\n" minimum(obj_blockprox)

# ADMM
#=
We note that since in the objective function of ADMM, the regularizer
of each edge is used twice. We need to use λ / 2 as the regularization
parameter to make sure the objective function is the same.
=#
# β comes from the code of the network lasso paper.
β = 1e-4 + sqrt(λ / 2)
maxiter_admm = ceil(2 * maxiter / 4 / ne(g)) |> Int
x_admm, obj_admm = admm(g, trainX, trainY, λ / 2, β, initial_x, loss_housing_cvx;
                        maxiter=maxiter_admm, disp_freq=disp_freq, wt=wts)
comms_admm = 4 * ne(g) * ones(maxiter_admm)

@printf "     ADMM: %+10.7e\n" obj_admm[end]

# Proximal Average
maxiter_prox_ave = ceil(2 * maxiter / 2 / ne(g)) |> Int
x_proxavg, obj_proxavg = proxavg(g, trainX, trainY, λ, initial_x, ss_const_housing,
        grad_housing, pair_prox2, loss_housing, reg2; maxiter=maxiter_prox_ave,
        disp_freq=disp_freq, wt=wts)
comms_proxavg = 2 * ne(g) * ones(maxiter_prox_ave)
@printf "  ProxAvg: %+10.7e\n" obj_proxavg[end]

# Decentralized subgradient descent
maxiter_dgd = ceil(2 * maxiter / 2 / ne(g)) |> Int
x_dgd, obj_dgd = dgd(g, trainX, trainY, λ/2, initial_x_dgd, ss_const_housing_dgd,
        grad_housing_dgd, loss_housing_dgd, reg2; maxiter=maxiter_dgd,
        disp_freq=disp_freq, wt=wts)
comms_dgd = 2 * ne(g) * ones(maxiter_dgd)
@printf "      DGD: %+10.7e\n" obj_dgd[end]

# Walkman
maxiter_walkman = ceil(2 * maxiter / num_nodes) |> Int
# Note that we are indeed solving the original problem divided by num_nodes.
x_walkman, obj_walkman = walkman(g, trainX, trainY, λ/num_nodes, initial_x_dgd,
        10000, grad_housing_walkman, loss_housing_walkman, reg2_walkman, prox2;
        maxiter=maxiter_walkman, disp_freq=disp_freq)
# To compare with other methods, we need to multiply the objective function
# value by num_nodes.
obj_walkman = obj_walkman .* num_nodes
comms_walkman = num_nodes * ones(maxiter_walkman)

# Compute the objective value at the initial point, i.e., zeros.
initobj = [loss_housing(trainX[i], trainY[i], zeros(dim)) for i = 1:num_nodes] |> sum

# Save the results.
# @save "housing.jld2" x_cvx optval x_blockprox obj_blockprox x_admm obj_admm x_proxavg obj_proxavg x_dgd obj_dgd x_walkman obj_walkman initobj comms_blockprox comms_admm comms_proxavg comms_dgd
# @load "housing.jld2" x_cvx optval x_blockprox obj_blockprox x_admm obj_admm x_proxavg obj_proxavg x_dgd obj_dgd x_walkman obj_walkman initobj comms_blockprox comms_admm comms_proxavg comms_dgd

# Plot
# Block proximal
plot([0; cumsum(comms_blockprox)], [initobj; obj_blockprox] .- optval,
     dpi=300, label="RandomEdge", color=:blue, alpha=0.6, linestyle=:solid, linewidth=2,
     xlabel="Communication", ylabel=L"H(x) - H^*", yscale=:log10, 
     legend=:best, tickfontsize=12, guidefontsize=16, titlefontsize=20,
     legendfontsize=14)
# ADMM
plot!([0; cumsum(comms_admm)], [initobj; obj_admm] .- optval, label="ADMM", color=:red, alpha=.6, marker=:utriangle, linewidth=2)
# Proximal Average
plot!([0; cumsum(comms_proxavg)], [initobj; obj_proxavg] .- optval, label="proxavg", color=:green, alpha=.6, marker=:xcross, linewidth=2)
# Decentralized subgradient descent
plot!([0; cumsum(comms_dgd)], [initobj; obj_dgd] .- optval, label="DSGD", color=:purple, alpha=.6, marker=:diamond, linewidth=2)
# Walkman
plot!([0; cumsum(comms_walkman)], [initobj; obj_walkman] .- optval, label="Walkman", color=:inferno, alpha=.6, linestyle=:dash, linewidth=2)
title!("Housing dataset")
xlims!(0, 2 * maxiter) |> display
# savefig("housing.png")