using Graphs
using Printf
using Random
using LinearAlgebra
using Plots
using Convex, SCS
using LaTeXStrings
using Dates
using JLD2
using StatsPlots
using Statistics
using Zygote
using StatsBase

"""
    g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges = gen_graph(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node, gen_truex,
        gen_data; noise=1e-2, same_prob=0.5, diff_prob=0.01)

Generate a graph.

# Arguments
## Input:
    - `num_group::Int`: the number of groups in the graph.
    - `num_nodes_group::Vector{<:Int}`: a vector of numbers of nodes in groups.
    - `dim::Int`: the dimension of the variable.
    - `num_training_node::Vector{<:Int}`: a vector of numbers of training data in nodes.
    - `num_testing_node::Vector{<:Int}`: a vector of numbers of testing data in nodes.
    - `gen_truex::Function`: a function for generating the true solution:
        `x_true = gen_truex(num_group::Int, dim::Int)`
    - `gen_data::Function`: a function for generating the training/testing datasets.
        `X, Y = gen_data(num::Int, dim::Int, x_true::Vector{<:Real}, noise::Real)`
    - `noise::Real`: the noise, default 1e-2.
    - `same_prob::Real`: the probability for two nodes in the same group to be
        connected by an edge, default 0.5.
    - `diff_prob::Real`: the probability for two nodes in different groups to
        be connected by an edge, default 0.01.

## Output:
    - `g::SimpleGraph`: the generated graph.
    - `x_true::Matrix{<:Real}`: the true x, a matrix, the i-th column is the true
        x of group i.
    - `trainX::Vector{Matrix{Float64}}`: the training dataset, the i-th element
        is the i-th node's data.
    - `trainY::Vector{Vector{Float64}}`: the training output, the i-th element
        is the i-th node's output.
    - `testX::Vector{Matrix{Float64}}`: the testing dataset, the i-th element
        is the i-th node's data.
    - `testY::Vector{Vector{Float64}}`: the testing output, the i-th element
        is the i-th node's output.
    - `groups::Vector{Int}`: the groups of nodes, the i-th element is the group
        id of the i-th node.
    - `degrees::Vector{Int}`: the degrees of nodes, the i-th element is the degree
        of the i-th node.
    - `correct_edges::Real`: the number of correct edges in the generated graph.
"""
function gen_graph(
    num_group::Int,
    num_nodes_group::Vector{<:Int},
    dim::Int,
    num_training_node::Vector{<:Int},
    num_testing_node::Vector{<:Int},
    gen_truex::Function,
    gen_data::Function;
    noise::Real=1e-2,
    same_prob::Real=0.5,
    diff_prob::Real=0.01,
)

    # The number of nodes in the graph.
    num_nodes = sum(num_nodes_group)
    # The cummulative sum of the number of nodes in each group.
    cum_sum_nodes = cumsum(num_nodes_group)
    # The graph.
    g = SimpleGraph(num_nodes)
    # The number of correct edges.
    correct_edges = 0

    # Generate groundtruth of variables.
    # Notice that we need to add 1 for the bias term.
    x_true = gen_truex(num_group, dim)

    # Generate the training/testing datasets for each node.
    # Training dataset.
    trainX = Vector{Matrix{Float64}}(undef, num_nodes)
    # Training labels.
    trainY = Vector{Vector{Float64}}(undef, num_nodes)
    # Testing dataset.
    testX = Vector{Matrix{Float64}}(undef, num_nodes)
    # Testing labels.
    testY = Vector{Vector{Float64}}(undef, num_nodes)

    # Groups of nodes.
    groups = zeros(Int, num_nodes)
    # Degrees of nodes.
    degrees = zeros(Int, num_nodes)

    #=
    ## Randomly add some edges to the graph
    For each pair of nodes (i, j), we add an edge with probability same_prob if
    the groups of i and j are the same and diff_prob otherwise.

    ## Generate true solutions for each group and local datasets for each node.
    To be more efficient, in the same loop, we also generate true solutions for
    each group and training/testing datasets for each node.
    =#
    for i = 1:num_nodes
        # The group the node i belongs to.
        groupᵢ = findfirst(cum_sum_nodes .>= i)

        # Generate the training dataset in this node.
        trainX[i], trainY[i] = gen_data(num_training_node[i], dim, x_true[:, groupᵢ], noise)

        # Generate the testing dataset in this node.
        testX[i], testY[i] = gen_data(num_testing_node[i], dim, x_true[:, groupᵢ], noise)

        # Update the group indicator of node i.
        groups[i] = groupᵢ

        for j = i+1:num_nodes
            # The group the node j belongs to.
            groupⱼ = findfirst(cum_sum_nodes .>= j)
            # If nodes i and j belong to the same group.
            if groupᵢ == groupⱼ
                # Add an edge with probability same_prob.
                if rand() >= 1 - same_prob
                    # Add an edge between i and j
                    # If the edge (i, j) already exists, `add_edge!` will return
                    # false, then skip.
                    if add_edge!(g, i, j)
                        # Increase the degree of i by 1.
                        degrees[i] += 1
                        # Increase the degree of j by 1.
                        degrees[j] += 1
                        # Count the correct edges
                        correct_edges += 1
                    end
                end
            else
                # If nodes i and j belong to different groups.
                # Add an edge with probability diff_prob.
                if rand() >= 1 - diff_prob
                    # Add an edge between i and j
                    # If the edge (i, j) already exists, `add_edge!` will return
                    # false, then skip.
                    if add_edge!(g, i, j)
                        # Increase the degree of i by 1.
                        degrees[i] += 1
                        # Increase the degree of j by 1.
                        degrees[j] += 1
                    end
                end
            end
        end
    end

    return g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges
end

"""
    x, obj, comms = blockprox(g, trainX, trainY, λ, degrees, initial_x, step_size,
        gradient, pair_prox, loss, reg; maxiter=1000, disp_freq=1, wt=ones(0, 0))

Perform blockprox on the graph `g` with training data `trainX` and labels `trainY`.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `degrees::Vector{<:Int}`: the degrees of nodes in the graph, the i-th
        element is the degree of the i-th node in the graph.
    - `initial_x::Function`: a function to initialize the variable, it should
        take two arguments, the dimension of the variable and the number of nodes
        in the graph, and return a matrix of size (dim, num_nodes).
            `x = initial_x(dim::Integer, num_nodes::Integer)`
    - `step_size::Function`: a function to compute the stepsize of gradient descent,
        it should take one argument, the iteration number, and return a real number.
            `ss = step_size(iter::Integer)`
    - `gradient::Function`: a function to compute the gradient of the local
        objective function, it should take three arguments, the training data,
        the training label, and the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `pair_prox::Function`: a function to compute the proximal operator
        of the pairwise regularization term, it should take four arguments, the
        variable of the current node, the variable of the neighbor node, the
        stepsize of proximal step, and the regularization parameter, and return
        a vector.
            `x_i = pair_prox(newx[:, i], z, ss_prox, λ * weights[i, neigh])`
    - `loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = loss(trainX[i], trainY[i], x[:, i])`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[:, i], x[:, neigh])`
    - `maxiter::Int`: the maximum number of iterations, default 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Matrix{<:Real}`: the final variable, a matrix of size (dim, num_nodes),
        the i-th column is the variable of the i-th node in the graph.
    - `obj::Vector{<:Real}`: the objective values at each iteration, a vector of
        size (maxiter,).
    - `comms::Vector{<:Real}`: the number of communications at each iteration,
        a vector of size (maxiter,).
"""
function blockprox(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    degrees::Vector{<:Int},
    initial_x::Function,
    step_size::Function,
    gradient::Function,
    pair_prox::Function,
    loss::Function,
    reg::Function;
    maxiter::Int=1000,
    disp_freq::Int=1, 
    wt::Matrix{<:Real}=ones(0, 0),
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end

    # The number of features.
    dim = size(trainX[1], 2)
    # The number of edges in the graph.
    num_edges = ne(g)
    # Neighbor list.
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    # Initialize the variables.
    x = initial_x(dim, num_nodes)
    newx = zeros(dim, num_nodes)
    # An averaged variable for output.
    xbar = deepcopy(x)

    # The objective values.
    obj = zeros(maxiter)
    # The communications.
    comms = zeros(maxiter)

    for iter = 1:maxiter
        # Stepsize of gradient descent.
        ss = step_size(iter)
        # Stepsize of proximal step.
        ss_prox = ss * num_edges

        # Gradient descent.
        Threads.@threads for i = 1:num_nodes
            # Compute the gradient of the objective function.
            grad = gradient(trainX[i], trainY[i], x[:, i])
            # Gradient descent step.
            newx[:, i] = x[:, i] - ss * grad
        end

        # The communications in this iteration.
        comms_iter = zeros(num_nodes)

        # Proximal update.
        Threads.@threads for i = 1:num_nodes
            # With probability degrees[i] / num_edges, the node remains the same.
            # If the degree of the node is 0 (i.e., it is isolated), the node 
            # remains the same.
            if rand() >= degrees[i] / num_edges || degrees[i] == 0
                # Do nothing.
                x[:, i] = newx[:, i]
            else
                # Increase the communication counter.
                comms_iter[i] = 1
                # Randomly select a neighbor.
                neigh = rand(neighbors_[i], 1)[1]
                # Compute the proximal operator.
                # The value of the variable of the neighbor.
                z = newx[:, neigh]

                # Proximal update.
                x[:, i] = pair_prox(newx[:, i], z, ss_prox, λ * weights[i, neigh])
            end
        end
        # Update the communications.
        comms[iter] = sum(comms_iter)
        # Update the averaged variable.
        xbar = ((iter - 1) * xbar + x) / iter

        # # Use the averaged variable to compute the objective value.
        # for i = 1:num_nodes
        #     obj[iter] += loss(trainX[i], trainY[i], xbar[:, i])
        #     reg_ = [weights[i, neigh] * reg(xbar[:, i], xbar[:, neigh]) for neigh in neighbors_[i]] |> sum
        #     obj[iter] += λ * reg_ / 2
        # end

        # Compute the objective value.
        for i = 1:num_nodes
            obj[iter] += loss(trainX[i], trainY[i], x[:, i])
            reg_ = [weights[i, neigh] * reg(x[:, i], x[:, neigh]) for neigh in neighbors_[i]] |> sum
            obj[iter] += λ * reg_ / 2
        end

        # Information display.
        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end
    end

    return x, obj, comms
end

"""
    x, optval = whole_cvx(g, trainX, trainY, λ, loss, reg; wt=ones(0, 0))

Call Convex.jl to solve the whole convex optimization on the graph `g` with 
training data `trainX` and labels `trainY`.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = loss(trainX[i], trainY[i], x[(i-1)*dim+1:i*dim])`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[(i-1)*dim+1:i*dim], x[(neigh-1)*dim+1:neigh*dim])`
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Vector{<:Real}`: the final variable, a vector of size (dim * num_nodes),
        the concatenation of the variables of all nodes in the graph.
    - `optval::Real`: the optimal value of the objective function.
"""
function whole_cvx(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    loss::Function,
    reg::Function;
    wt::Matrix{<:Real}=ones(0, 0)
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end
    # The number of features.
    dim = size(trainX[1], 2)

    # Neighbor list.
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    # Variable
    x = Variable(dim * num_nodes)

    obj = 0.0
    for i = 1:num_nodes
        obj += loss(trainX[i], trainY[i], x[(i-1)*dim+1:i*dim])
        reg_ = [weights[i, neigh] * reg(x[(i-1)*dim+1:i*dim], x[(neigh-1)*dim+1:neigh*dim]) for neigh in neighbors_[i]] |> sum
        obj += λ * reg_ / 2
    end

    prob = minimize(obj)
    solve!(prob, SCS.Optimizer, silent=true)

    x_ = evaluate(x)

    return x_, prob.optval
end

"""
    x, obj = admm(g, trainX, trainY, λ, β, initial_x, local_loss; maxiter=1000,
        disp_freq=1, wt=ones(0, 0))

Perform ADMM to solve the network lasso problem.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `β::Real`: the ADMM parameter, should be larger than 0.
    - `initial_x::Function`: a function to initialize the variable, it should
        take two arguments, the dimension of the variable and the number of nodes
        in the graph, and return a matrix of size (dim, num_nodes).
            `x = initial_x(dim::Integer, num_nodes::Integer)`
    - `local_loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = local_loss(trainX[i], trainY[i], x[:, i])`
    - `maxiter::Int`: the maximum number of iterations, default 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Matrix{<:Real}`: the final variable, a matrix of size (dim, num_nodes),
        the i-th column is the variable of the i-th node in the graph.
    - `obj::Vector{<:Real}`: the objective values at each iteration, a vector of
        size (maxiter,).
"""
function admm(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    β::Real,
    initial_x::Function,
    local_loss::Function;
    maxiter::Int=1000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0)
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end
    # The number of features.
    dim = size(trainX[1], 2)
    # Neighbor list
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    # Initialize the variables.
    x = initial_x(dim, num_nodes)
    z = [Dict(neigh => zeros(dim) for neigh in neighbors_[i]) for i = 1:num_nodes]
    u = [Dict(neigh => zeros(dim) for neigh in neighbors_[i]) for i = 1:num_nodes]

    # The objective values.
    obj = zeros(maxiter)

    for iter = 1:maxiter
        loss_ = zeros(num_nodes)
        # x update.
        for i = 1:num_nodes
            x_ = Variable(dim)
            loss = local_loss(trainX[i], trainY[i], x_)
            prox_term = 0
            for neigh in neighbors_[i]
                prox_term += β / 2 * sumsquares(x_ - z[i][neigh] + u[i][neigh])
            end

            problem = minimize(loss + prox_term)
            solve!(problem, SCS.Optimizer; silent=true)
            x[:, i] = evaluate(x_)
            loss_[i] = evaluate(loss)
        end

        # z update.
        for i = 1:num_nodes
            for neigh in neighbors_[i]
                xu_i = x[:, i] + u[i][neigh]
                xu_j = x[:, neigh] + u[neigh][i]
                θ = max(1 - λ * weights[i, neigh] / (β * norm(xu_i - xu_j, 2) + 0.000001), 0.5)
                z[i][neigh] = θ * xu_i + (1 - θ) * xu_j
            end
        end

        # u udpate.
        for i = 1:num_nodes
            for neigh in neighbors_[i]
                u[i][neigh] += x[:, i] - z[i][neigh]
            end
        end

        for i = 1:num_nodes
            obj[iter] += loss_[i]
            for neigh in neighbors_[i]
                obj[iter] += λ * weights[i, neigh] * norm(x[:, i] - x[:, neigh], 2)
            end
        end

        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end
    end

    return x, obj
end

"""
    x, obj = dgd(g, trainX, trainY, λ, initial_x, step_size, gradient,
        local_loss, reg; maxiter=1000, disp_freq=1, wt=ones(0, 0))

Perform decentralized gradient descent on the graph `g` with training data `trainX`
and labels `trainY`.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `initial_x::Function`: a function to initialize the variable, it should
        take two arguments, the dimension of the variable and the number of nodes
        in the graph, and return a matrix of size (dim, num_nodes).
            `x = initial_x(dim::Integer, num_nodes::Integer)`
    - `step_size::Function`: a function to compute the stepsize of gradient descent,
        it should take one argument, the iteration number, and return a real number.
            `ss = step_size(iter::Integer)`
    - `gradient::Function`: a function to compute the gradient of the local
        objective function, it should take three arguments, the training data,
        the training label, and the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `local_loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = local_loss(trainX[i], trainY[i], x[:, i])`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[(i-1)*dim+1:i*dim], x[(neigh-1)*dim+1:neigh*dim])`
    - `maxiter::Int`: the maximum number of iterations, default 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Matrix{<:Real}`: the final variable, a matrix of size (dim, num_nodes),
        the i-th column is the variable of the i-th node in the graph.
    - `obj::Vector{<:Real}`: the objective values at each iteration, a vector of
        size (maxiter,).
"""
function dgd(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    initial_x::Function,
    step_size::Function,
    gradient::Function,
    local_loss::Function,
    reg::Function;
    maxiter::Int=1000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0)
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end
    # The number of features.
    dim = size(trainX[1], 2)
    # Neighbor list
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    #=
    Construct mixing matrix based on the graph g and Metropolis-Hastings
    Weights:
    W[i, j] = 1 / max(degree(i), degree(j)) if i and j are neighbors.
    W[i, i] = 1 - sum(W[i, :])
    W[i, j] = 0 otherwise. 
    =#
    # Declare the mixing matrix.
    W = zeros(num_nodes, num_nodes)
    # Construct the mixing matrix.
    for i = 1:num_nodes
        for j in neighbors_[i]
            W[i, j] = 1 / max(length(neighbors_[i]), length(neighbors_[j]))
        end
        W[i, i] = 1 - sum(W[i, :])
    end

    # Initialize the variables.
    x = initial_x(dim, num_nodes)
    newx = initial_x(dim, num_nodes)
    # An averaged variable for output.
    xbar = deepcopy(x)

    # The objective values.
    obj = zeros(maxiter)

    for iter = 1:maxiter
        # Local gradient descent.
        # Stepsize of gradient descent.
        ss = step_size(iter)
        Threads.@threads for i = 1:num_nodes
            # Local gradient
            grad = gradient(trainX[i], trainY[i], x[:, i], i, g, λ, reg)
            # Mixing neighbors' information and graadient descent.
            newx[:, i] = sum([W[i, neigh] * x[:, neigh] for neigh in neighbors_[i]]) + W[i, i] * x[:, i] - ss * grad
        end

        x = newx
        # Update the averaged variable.
        xbar = ((iter - 1) * xbar + x) / iter

        # Use the averaged variable to compute the objective value.
        for i = 1:num_nodes
            obj[iter] += local_loss(trainX[i], trainY[i], xbar[:, i], i, g, λ, reg)
        end

        # Information display.
        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end
    end

    return x, obj
end

"""
    x, obj = proxavg(g, trainX, trainY, λ, initial_x, step_size, gradient, pair_prox,
        local_loss, reg; maxiter=1000, disp_freq=1, wt=ones(0, 0))

Perform proximal average algorithm on the graph `g` with training data `trainX` and
labels `trainY`.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `initial_x::Function`: a function to initialize the variable, it should
        take two arguments, the dimension of the variable and the number of nodes
        in the graph, and return a matrix of size (dim, num_nodes).
            `x = initial_x(dim::Integer, num_nodes::Integer)`
    - `step_size::Function`: a function to compute the stepsize of gradient descent,
        it should take one argument, the iteration number, and return a real number.
            `ss = step_size(iter::Integer)`
    - `gradient::Function`: a function to compute the gradient of the local
        objective function, it should take three arguments, the training data,
        the training label, and the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `pair_prox::Function`: a function to compute the proximal operator
        of the pairwise regularization term, it should take four arguments, the
        variable of the current node, the variable of the neighbor node, the
        stepsize of proximal step, and the regularization parameter, and return
        a vector.
            `x_i = pair_prox(newx[:, i], z, ss_prox, λ * weights[i, neigh])`
    - `local_loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = local_loss(trainX[i], trainY[i], x[:, i])`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[(i-1)*dim+1:i*dim], x[(neigh-1)*dim+1:neigh*dim])`
    - `maxiter::Int`: the maximum number of iterations, default 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Matrix{<:Real}`: the final variable, a matrix of size (dim, num_nodes),
        the i-th column is the variable of the i-th node in the graph.
    - `obj::Vector{<:Real}`: the objective values at each iteration, a vector of
        size (maxiter,).
"""
function proxavg(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    initial_x::Function,
    step_size::Function,
    gradient::Function,
    pair_prox::Function,
    local_loss::Function,
    reg::Function;
    maxiter::Int=1000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0)
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end
    # The number of edges in the graph.
    num_edges = ne(g)
    # The number of features.
    dim = size(trainX[1], 2)
    # Neighbor list
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    # Initialize the variables.
    x = initial_x(dim, num_nodes)
    newx = zeros(dim, num_nodes)

    # The objective values.
    obj = zeros(maxiter)

    for iter = 1:maxiter
        # Local gradient descent.
        # Stepsize of gradient descent.
        ss = step_size(iter)
        ss_prox = num_edges
        grad = hcat([gradient(trainX[i], trainY[i], x[:, i]) for i = 1:num_nodes]...)
        newx = x - ss * grad

        Threads.@threads for i = 1:num_nodes
            x_ = zeros(dim)
            for neigh in neighbors_[i]
                # The value of the variable of the neighbor.
                z = newx[:, neigh]
                x_ += pair_prox(newx[:, i], z, ss_prox, λ * weights[i, neigh])
            end
            x[:, i] = (x_ + (num_edges - length(neighbors_[i])) * newx[:, i]) / num_edges
        end

        # Compute the objective value.
        for i = 1:num_nodes
            obj[iter] += local_loss(trainX[i], trainY[i], x[:, i])
            reg_ = [weights[i, neigh] * reg(x[:, i], x[:, neigh]) for neigh in neighbors_[i]] |> sum
            obj[iter] += λ * reg_ / 2
        end

        # Information display.
        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end
    end

    return x, obj
end

function walkman(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    initial_x::Function,
    β::Real,
    gradient::Function,
    loss::Function,
    reg::Function,
    prox::Function;
    maxiter::Int=1000,
    disp_freq::Int=1,
)
    # The number of nodes in the graph.
    n = nv(g)
    # The number of features.
    dim = size(trainX[1], 2)
    # Neighbor list
    neighbors_ = [neighbors(g, i) for i = 1:n]
    # Degrees
    degrees = [degree(g, i) for i = 1:n]
    # Initialize the variables.
    x = initial_x(dim, n)
    y = x
    z = x
    x̄ = x

    # The adjacency matrix of g.
    A = adjacency_matrix(g)
    # Construct the transition probability matrix.
    P = A ./ (maximum(degrees)) |> Matrix

    # The objective values.
    obj = zeros(maxiter)

    # Current node, randomly selected.
    iₖ = rand(1:n, 1)[1]

    for iter = 1:maxiter
        # Update (10a)
        x[:, iₖ] = prox(x̄[:, iₖ], g, λ / β)

        # Store something that will be used in (10d).
        diff_yz = y[:, iₖ] - z[:, iₖ] / β

        # Update (10b')
        y[:, iₖ] = x[:, iₖ] + 1/β * z[:, iₖ] - 1/β * gradient(trainX[iₖ], trainY[iₖ], y[:, iₖ], iₖ)

        # Update (10c)
        z[:, iₖ] = z[:, iₖ] + β * (x[:, iₖ] - y[:, iₖ])

        # Update (10d)
        x̄[:, iₖ] = x̄[:, iₖ] + 1/n * (y[:, iₖ] - z[:, iₖ] / β) - 1/n * diff_yz

        # Next agent.
        # iₖ₊₁ = rand(neighbors_[iₖ], 1)[1]
        w_ = P[iₖ, neighbors_[iₖ]] ./ sum(P[iₖ, neighbors_[iₖ]])
        iₖ₊₁ = sample(neighbors_[iₖ], Weights(w_))

        # Update variable in the next agent.
        x̄[:, iₖ₊₁] = x̄[:, iₖ]

        # Compute all local loss function values in every node.
        for i in 1:n
            obj[iter] += loss(trainX[i], trainY[i], x[:, i], i) / n
        end
        # Compute the value of regularizer.
        obj[iter] += reg(x[:, iₖ], g, λ)

        # Update agent id.
        iₖ = iₖ₊₁

        # Information display.
        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end
    end

    return x, obj
end

"""
    x, obj, comms = scaffnew(g, trainX, trainY, λ, initial_x, ss_grad_desc,
        ss_prox, gradient, local_loss, reg; p=1 / 10, maxiter=1000,
        disp_freq=1, wt=ones(0, 0))

Perform the Scaffnew algorithm on the graph `g` with training data `trainX` and
labels `trainY`.

# Arguments
## Input:
    - `g::SimpleGraph`: the communication network and the underlying graph of the
        regularizers.
    - `trainX::Vector`: the training dataset, the i-th element is the training
        data of the i-th node in the graph.
    - `trainY::Vector`: the training labels, the i-th element is the training
        labels of the i-th node in the graph.
    - `λ::Real`: the regularization parameter.
    - `initial_x::Function`: a function to initialize the variable, it should
        take two arguments, the dimension of the variable and the number of nodes
        in the graph, and return a matrix of size (dim, num_nodes).
            `x = initial_x(dim::Integer, num_nodes::Integer)`
    - `ss_grad_desc::Function`: a function to compute the stepsize of gradient descent,
        it should take one argument, the iteration number, and return a real number.
            `ss = ss_grad_desc(iter::Integer)`
    - `ss_prox::Function`: a function to compute the stepsize of proximal step,
        it should take one argument, the iteration number, and return a real number.
            `τ = ss_prox(iter::Integer)`
    - `gradient::Function`: a function to compute the gradient of the local
        objective function, it should take three arguments, the training data,
        the training label, and the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `local_loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = local_loss(trainX[i], trainY[i], x[:, i])`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[:, i], x[:, neigh])`
    - `p::Real`: the probability of performing the proximal step, default 1 / 10.
    - `maxiter::Int`: the maximum number of iterations, default 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.

## Output:
    - `x::Matrix{<:Real}`: the final variable, a matrix of size (dim, num_nodes),
        the i-th column is the variable of the i-th node in the graph.
    - `obj::Vector{<:Real}`: the objective values at each iteration, a vector of
        size (maxiter,).
    - `comms::Vector{<:Real}`: the number of communications at each iteration,
        a vector of size (maxiter,).
"""
function scaffnew(
    g::SimpleGraph,
    trainX::Vector,
    trainY::Vector,
    λ::Real,
    initial_x::Function, 
    ss_grad_desc::Function, 
    ss_prox::Function, 
    gradient::Function, 
    local_loss::Function, 
    reg::Function;
    p::Real=1 / 10,
    maxiter::Int=1000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0)
)

    # The number of nodes in the graph.
    num_nodes = nv(g)
    # Checking the weights.
    # If it is the default value, we set it to ones, i.e., the problem
    # does not have weights.
    if size(wt) == (0, 0)
        weights = ones(num_nodes, num_nodes)
    else
        weights = wt
    end
    # The number of edges.
    num_edges = ne(g)
    # The number of features.
    dim = size(trainX[1], 2)
    # Neighbor list
    neighbors_ = [neighbors(g, i) for i = 1:num_nodes]

    #=
    Construct mixing matrix based on the graph g and Metropolis-Hastings
    Weights:
    W[i, j] = 1 / max(degree(i), degree(j)) if i and j are neighbors.
    W[i, i] = 1 - sum(W[i, :])
    W[i, j] = 0 otherwise. 
    =#
    # Declare the mixing matrix.
    W = zeros(num_nodes, num_nodes)
    # Construct the mixing matrix.
    for i = 1:num_nodes
        for j in neighbors_[i]
            W[i, j] = 1 / max(length(neighbors_[i]), length(neighbors_[j]))
        end
        W[i, i] = 1 - sum(W[i, :])
    end

    # Initialize the variables.
    x = initial_x(dim, num_nodes)
    h = zeros(dim, num_nodes)
    newx = zeros(dim, num_nodes)

    # The objective values.
    obj = zeros(maxiter)
    # The communications.
    comms = zeros(maxiter)

    iter = 1

    while iter <= maxiter
        # Local gradient descent.
        # Stepsize of gradient descent.
        ss = ss_grad_desc(iter)
        τ = ss_prox(iter)

        # Flip a coin to decide whether to skip the prox.
        θ = rand() <= p ? 1 : 0

        Threads.@threads for i = 1:num_nodes
            # Local gradient
            grad = gradient(trainX[i], trainY[i], x[:, i])
            # Mixing neighbors' information and graadient descent.
            newx[:, i] = x[:, i] - ss * (grad - h[:, i])
        end

        if θ == 1
            Threads.@threads for i = 1:num_nodes
                x[:, i] = (1 - ss * τ / p) * newx[:, i] + ss * τ / p * (sum([W[i, j] * newx[:, j] for j in neighbors_[i]]) + W[i, i] * newx[:, i])
                h[:, i] = h[:, i] + p / ss * (x[:, i] - newx[:, i])
            end
            comms[iter] = 2 * num_edges
        else
            x = newx
        end

        # Compute the objective value.
        for i = 1:num_nodes
            obj[iter] += local_loss(trainX[i], trainY[i], x[:, i])
            reg_ = [weights[i, neigh] * reg(x[:, i], x[:, neigh]) for neigh in neighbors_[i]] |> sum
            obj[iter] += λ * reg_ / 2
        end

        # Information display.
        if iter % disp_freq == 0
            @printf " %6d|%+10.7e|\n" iter obj[iter]
        end

        iter += 1
    end

    return x, obj, comms
end

#===============================================================================
        Functions declaration starts here.
===============================================================================#

"""
    inti_x = initial_x(dim::Integer, num_nodes::Integer)
"""
function initial_x(dim::Integer, num_nodes::Integer)
    return zeros(dim, num_nodes)
end

"""
    x_true = gen_truex(num_group::Integer, dim::Integer)
"""
function gen_truex(num_group::Integer, dim::Integer)
    x_true = randn(dim + 1, num_group)
    # x_true = 2 .* randn(dim + 1, num_group) .+ 3
    return x_true
end

"""
    X, Y = gen_data_LS(num::Integer, dim::Integer, x_true::Vector, noise::Float64)
"""
function gen_data_LS(num::Integer, dim::Integer, x_true::Vector, noise::Float64)
    X = randn(num, dim + 1)
    # The bias term.
    X[:, end] .= 1

    # The true labels.
    Y = X * x_true + noise * randn(num) |> vec

    return X, Y
end

"""
    ss = ss_ook(iter::Integer)
"""
function ss_ook(iter::Integer)
    # Stepsize, one over k (ook).
    return 1e-2 / iter
end

"""
    ss = ss_ooqk(iter::Integer)
"""
function ss_ooqk(iter::Integer)
    # Stepsize, one over sqrt(k) (ooqk).
    return 1e-2 / sqrt(iter+1)
end

"""
    ss = ss_gap(iter::Integer; gap::Integer=180, init::Float64=2e-2, desc::Real=1.5)
"""
function ss_gap(iter::Integer; gap::Integer=180, init::Float64=2e-2, desc::Real=1.5)
    return max(init / desc^floor(iter / gap), 1e-5)
end

"""
    ss = ss_const(iter::Integer)
"""
function ss_const(iter::Integer)
    return 1e-2
end

"""
    grad = grad_LS(X::Matrix, Y::Vector, x::Vector)
"""
function grad_LS(X::Matrix, Y::Vector, x::Vector)
    return X' * (X * x - Y)
end

"""
    x = pair_prox2(x::Vector, z::Vector, ss::Real, λ::Real)
"""
function pair_prox2(x::Vector, z::Vector, ss::Real, λ::Real)
    norm_diff = norm(x - z, 2)
    if norm_diff <= 2 * ss * λ
        return (x + z) / 2
    else
        return x - ss * λ * (x - z) / norm_diff
    end
end

"""
    y = pair_prox1(x::Vector, z::Vector, ss::Real, λ::Real)
"""
function pair_prox1(x::Vector, z::Vector, ss::Real, λ::Real)
    y = sign.(x - z) .* max.(abs.(x - z) .- 2 * ss * λ, 0)
    return (y + x + z) / 2
end

function reg2_walkman(
    x::Vector,
    g::SimpleGraph,
    λ::Real,
)
    # The number of nodes in g.
    num_nodes = nv(g)

    # The dimension of each local variable.
    dim = length(x) / num_nodes |> Int

    val = 0
    for edge in edges(g)
        src = edge.src
        dst = edge.dst
        val += λ * norm(x[(src - 1) * dim + 1:src * dim] - x[(dst - 1) * dim + 1:dst * dim], 2)
    end

    return val
end

function prox2(
    x::Vector,
    g::SimpleGraph,
    λ::Real,
)
    # The number of nodes in g.
    num_nodes = nv(g)

    # The dimension of each local variable.
    dim = length(x) / num_nodes |> Int

    # The convex variable.
    z = Variable(length(x))

    # The objective value.
    obj = 1/2 * sumsquares(x - z)
    # For each edge, compute the 2 norm.
    for edge in edges(g)
        src = edge.src
        dst = edge.dst
        obj += λ * norm(z[(src - 1) * dim + 1:src * dim] - z[(dst - 1) * dim + 1:dst * dim], 2)
    end

    # Minimize.
    prob = minimize(obj)
    solve!(prob, SCS.Optimizer, silent=true)

    z_ = evaluate(z)

    return z_
end

function reg1_walkman(
    x::Vector,
    g::SimpleGraph,
    λ::Real,
)
    # The number of nodes in g.
    num_nodes = nv(g)

    # The dimension of each local variable.
    dim = length(x) / num_nodes |> Int

    val = 0
    for edge in edges(g)
        src = edge.src
        dst = edge.dst
        val += λ * norm(x[(src - 1) * dim + 1:src * dim] - x[(dst - 1) * dim + 1:dst * dim], 1)
    end

    return val
end

function prox1(
    x::Vector,
    g::SimpleGraph,
    λ::Real,
)
    # The number of nodes in g.
    num_nodes = nv(g)

    # The dimension of each local variable.
    dim = length(x) / num_nodes |> Int

    # The convex variable.
    z = Variable(length(x))

    # The objective value.
    obj = 1/2 * sumsquares(x - z)
    # For each edge, compute the 2 norm.
    for edge in edges(g)
        src = edge.src
        dst = edge.dst
        obj += λ * norm(z[(src - 1) * dim + 1:src * dim] - z[(dst - 1) * dim + 1:dst * dim], 1)
    end

    # Minimize.
    prob = minimize(obj)
    solve!(prob, SCS.Optimizer, silent=true)

    z_ = evaluate(z)

    return z_
end

"""
    lossval = loss_LS(X::Matrix, Y::Vector, x::Vector)
"""
function loss_LS(X::Matrix, Y::Vector, x::Vector)
    return norm(X * x - Y, 2)^2 / 2
end

"""
    lossval = loss_LS_cvx(X::Matrix, Y::Vector, x<:Convex.AbstractExpr)
"""
function loss_LS_cvx(X::Matrix, Y::Vector, x::T) where {T<:Convex.AbstractExpr}
    return sumsquares(X * x - Y) / 2
end

initial_x_dgd(dim::Int, num_nodes::Int) = zeros(dim * num_nodes, num_nodes)

function loss_LS_dgd(
    X::Matrix{<:Real},
    Y::Vector{<:Real},
    x::Vector{<:Real},
    i::Int,
    g::SimpleGraph,
    λ::Real,
    reg::Function,
)
    # Dimension.
    dim = size(X, 2)
    # Local loss function value.
    val = norm(X * x[(i - 1) * dim + 1 : i * dim] - Y, 2) ^ 2 / 2

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

function loss_LS_walkman(
    X::Matrix{<:Real},
    Y::Vector{<:Real},
    x::Vector{<:Real},
    i::Int,
)
    # Dimension.
    dim = size(X, 2)
    # Local loss function value.
    val = norm(X * x[(i - 1) * dim + 1 : i * dim] - Y, 2) ^ 2 / 2
    
    return val
end

function grad_LS_dgd(
    X::Matrix{<:Real},
    Y::Vector{<:Real},
    x::Vector{<:Real},
    i::Int,
    g::SimpleGraph,
    λ::Real,
    reg::Function,
)
    ∇ = gradient(x_ -> loss_LS_dgd(X, Y, x_, i, g, λ, reg), x)[1]
    return ∇ === nothing ? zero(x) : ∇
end

function grad_LS_walkman(
    X::Matrix{<:Real},
    Y::Vector{<:Real},
    x::Vector{<:Real},
    i::Int,
)
    ∇ = gradient(x_ -> loss_LS_walkman(X, Y, x_, i), x)[1]
    return ∇ === nothing ? zero(x) : ∇
end

"""
    regval = reg2_cvx(x::T, z::T) where {T<:Convex.AbstractExpr}
"""
function reg2_cvx(x::T, z::T) where {T<:Convex.AbstractExpr}
    return norm(x - z, 2)
end

"""
    regval = reg1_cvx(x::T, z::T) where {T<:Convex.AbstractExpr}
"""
function reg1_cvx(x::T, z::T) where {T<:Convex.AbstractExpr}
    return norm(x - z, 1)
end

"""
    regval = reg2(x::Vector, z::Vector)
"""
function reg2(x::Vector, z::Vector)
    return norm(x - z, 2)
end

"""
    regval = reg1(x::Vector, z::Vector)
"""
function reg1(x::Vector, z::Vector)
    return norm(x - z, 1)
end

# Other stepsize functions.
ss_gap2(iter::Integer) = ss_gap(iter; gap=80, init=1e-2)
ss_const2(iter::Integer) = 1e-3
ss_const3(iter::Integer) = 1e-5

#===============================================================================
        Functions of synthetic experiments.
===============================================================================#
"""
    init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox,
    x_admm, obj_admm, comms_admm, x_proxavg, obj_proxavg, comms_proxavg,
    x_dgd, obj_dgd, comms_dgd = run_syn_exp_LS_2norm(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node, λ,
        ss_blockprox, ss_dgd, ss_proxavg, β_walkman; same_prob=0.5, diff_prob=0.01,
        noise=1e-2, maxiter=10000, disp_freq=1, wt=ones(0, 0), loss=loss_LS,
        loss_dgd=loss_LS_dgd, loss_walkman=loss_LS_walkman, reg=reg2, 
        reg_walkman=reg2_walkman, loss_cvx=loss_LS_cvx,
        reg_cvx=reg2_cvx, gradient=grad_LS, grad_dgd=grad_LS_dgd,
        grad_walkman=grad_LS_walkman, pair_prox=pair_prox2, prox_walkman=prox2,
        gen_truex=gen_truex, gen_data=gen_data_LS, initial_x=initial_x,
        initial_x_dgd=initial_x_dgd, dispinfo=true,
        showfig=true, savefigure=false, savevar=false)

Run the synthetic experiment for the least squares problem with fused 2-norm regularization.

# Arguments
## Input:
    - `num_group::Int`: the number of groups in the graph.
    - `num_nodes_group::Vector{<:Int}`: the number of nodes in each group, a vector of size (num_group,).
    - `dim::Int`: the dimension of the data, the number of features.
    - `num_training_node::Vector{<:Int}`: the number of training nodes in each group,
        a vector of size (num_group,).
    - `num_testing_node::Vector{<:Int}`: the number of testing nodes in each group,
        a vector of size (num_group,).
    - `λ::Real`: the regularization parameter.
    - `ss_blockprox::Function`: the step size function for block proximal method.
    - `ss_dgd::Function`: the step size function for consensus optimization.
    - `ss_proxavg::Function`: the step size function for proximal average method.
    - `β_walkman::Real`: the step size for walkman.
    - `same_prob::Real`: the probability of having the same edge in the graph,
        default is 0.5.
    - `diff_prob::Real`: the probability of having a different edge in the graph,
        default is 0.01.
    - `noise::Real`: the noise level in the data generation, default is 1e-2.
    - `maxiter::Int`: the maximum number of iterations for the algorithms, default is 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default is 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.
    - `loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = loss(trainX[i], trainY[i], x[:, i])`
    - `loss_dgd::Function`: a function to compute the local loss of dgd, it should
        take seven arguments: the training data, the training label, the variable,
        the index of node, and the graph, λ, regularizer, and return a real number.
            `lossval = loss_dgd(X, Y, x, i, g, λ, reg)`
    - `loss_walkman::Function`: a function to compute the local loss of walkman,
        it should take four arguments: the training data, the training label,
        the variable, the index of node, and return a real number.
            `lossval = loss_walkman(X, Y, x, i)`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[:, i], x[:, neigh])`
    - `reg_walkman::Function`: a function to compute the whole regularizer, it should
        take three arguments, the variable of the current node, the graph, the
        regularization parameter, and return a real number.
            `regval = reg_walkman(x[:, i], g, λ)`
    - `loss_cvx::Function`: a function to compute the local loss for convex optimization,
        it should take three arguments, the training data, the training label, and
        the variable, and return a Convex.jl expression.
            `lossval = local_loss_cvx(trainX[i], trainY[i], x[:, i])`
    - `reg_cvx::Function`: a function to compute the regularization term for convex optimization,
        it should take two arguments, the variable of the current node and the variable
        of the neighbor node, and return a Convex.jl expression.
            `regval = reg_cvx(x[:, i], x[:, neigh])`
    - `gradient::Function`: a function to compute the gradient of the local loss,
        it should take three arguments, the training data, the training label, and
        the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `grad_dgd::Function`: a function to compute the gradient of the local loss for dgd,
        it should take seven arguments: the training data, the training label,
        the variable, the index of node, the graph, λ, regularizer, and return a vector.
            `grad = grad_dgd(X, Y, x, i, g, λ, reg)`
    - `grad_walkman::Function`: a function to compute the gradient of the local
        loss for walkman, it should take four arguments: the training data, the
        training label, the variable, the index of node, and return a vector.
            `grad = grad_walkman(X, Y, x, i)`
    - `pair_prox::Function`: a function to compute the proximal operator for the pair
        of variables, it should take three arguments, the variable of the current node,
        the variable of the neighbor node, and the step size, and return a vector.
            `y = pair_prox(x[:, i], x[:, neigh], ss, λ)`
    - `prox_walkman::Function`: a function to compute the proximal operator of
        the whole regulaizer, it should take three arguments, the variable,
        the graph, and the regulaization parameter.
            `y = prox_walkman(x[:, i], g, λ)`
    - `gen_truex::Function`: a function to generate the true variable, it should
        take two arguments, the number of groups and the dimension, and return a vector.
            `x_true = gen_truex(num_group, dim)`
    - `gen_data::Function`: a function to generate the training data and labels,
        it should take four arguments, the number of nodes, the dimension, the true variable,
        and the noise level, and return a tuple of training data and labels.
            `trainX, trainY = gen_data(num_nodes, dim, x_true, noise)`
    - `initial_x::Function`: a function to initialize the variable, it should take
        two arguments, the dimension and the number of nodes, and return a matrix.
            `x = initial_x(dim, num_nodes)`
    - `initial_x_dgd::Function`: a function to initialize the variable for dgd, it should take
        two arguments, the dimension and the number of nodes, and return a matrix.
            `x = initial_x_dgd(dim, num_nodes)`
    - `dispinfo::Bool`: whether to display the information during the experiment,
        default is true.
    - `showfig::Bool`: whether to show the figure of the results, default is true.
    - `savefigure::Bool`: whether to save the figure of the results, default is false.
    - `savevar::Bool`: whether to save the variables of the results, default is false.

## Output:
    - `init_obj<:Real`: the initial objective value, a vector of size (1,).
    - `x_cvx::Vector{<:Real}`: the optimal solution from convex optimization.
    - `optval::Real`: the optimal value from convex optimization.
    - `x_block_prox::Matrix{<:Real}`: the solution from block proximal method.
    - `obj_block_prox::Vector{<:Real}`: the objective values from block proximal method.
    - `comms_block_prox::Vector{<:Real}`: the number of communications from block proximal method.
    - `x_proxavg::Matrix{<:Real}`: the solution from proximal average method.
    - `obj_proxavg::Vector{<:Real}`: the objective values from proximal average method.
    - `comms_proxavg::Vector{<:Real}`: the number of communications from proximal average method.
    - `x_dgd::Matrix{<:Real}`: the solution from decentralized subgradient descent method.
    - `obj_dgd::Vector{<:Real}`: the objective values from decentralized subgradient descent method.
    - `comms_dgd::Vector{<:Real}`: the number of communications from decentralized subgradient descent method.
    - `x_walkman::Matrix{<:Real}`: the solution from walkman.
    - `obj_walkman::Vector{<:Real}`: the objective values from walkman, note that this one is already multiplied by num_nodes.
"""
function run_syn_exp_LS_2norm(
    num_group::Int,
    num_nodes_group::Vector{<:Int},
    dim::Int,
    num_training_node::Vector{<:Int},
    num_testing_node::Vector{<:Int},
    λ::Real,
    ss_blockprox::Function,
    ss_dgd::Function,
    ss_proxavg::Function,
    β_walkman::Real;
    same_prob::Real=0.5,
    diff_prob::Real=0.01,
    noise::Real=1e-2,
    maxiter::Int=10000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0),
    loss::Function=loss_LS,
    loss_dgd::Function=loss_LS_dgd,
    loss_walkman::Function=loss_LS_walkman,
    reg::Function=reg2,
    reg_walkman::Function=reg2_walkman,
    loss_cvx::Function=loss_LS_cvx,
    reg_cvx::Function=reg2_cvx,
    gradient::Function=grad_LS,
    grad_dgd::Function=grad_LS_dgd,
    grad_walkman::Function=grad_LS_walkman,
    pair_prox::Function=pair_prox2,
    prox_walkman::Function=prox2,
    gen_truex::Function=gen_truex,
    gen_data::Function=gen_data_LS,
    initial_x::Function=initial_x,
    initial_x_dgd::Function=initial_x_dgd,
    dispinfo::Bool=true,
    showfig::Bool=true,
    savefigure::Bool=false,
    savevar::Bool=false,
)

    # Figure title.
    figtitle = "LS" * " with fused " * "2-norm regularization"
    # Figure name.
    figname = "LS_2_" * string(num_group) * "_" * Dates.format(now(), "yymmddHHMMSS") * ".png"
    # File name.
    savename = "LS_2_" * string(num_group) * "_" * Dates.format(now(), "yymmddHHMMSS") * ".jld2"

    # Generate the graph.
    g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges = gen_graph(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node, gen_truex,
        gen_data; noise=noise, same_prob=same_prob, diff_prob=diff_prob)
    
    # We need to avoid unconnected graph since our competing methods require
    # connectivity.
    while !is_connected(g)
        g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges = gen_graph(
            num_group, num_nodes_group, dim, num_training_node, num_testing_node, gen_truex,
            gen_data; noise=noise, same_prob=same_prob, diff_prob=diff_prob)
    end

    # Convex.jl
    x_cvx, optval = whole_cvx(g, trainX, trainY, λ, loss_cvx, reg_cvx; wt=wt)

    # Block Proximal
    x_block_prox, obj_block_prox, comms_block_prox = blockprox(g, trainX,
        trainY, λ, degrees, initial_x, ss_blockprox, gradient, pair_prox, loss,
        reg; maxiter=maxiter, disp_freq=disp_freq, wt=wt)

    # ADMM
    #=
    We note that since in the objective function of ADMM, the regularizer
    of each edge is used twice. We need to use λ / 2 as the regularization
    parameter to make sure the objective function is the same.
    =#
    # β comes from the code of the network lasso paper.
    β = 1e-4 + sqrt(λ / 2)
    maxiter_admm = ceil(15000 / 4 / ne(g)) |> Int
    x_admm, obj_admm = admm(g, trainX, trainY, λ / 2, β, initial_x, loss_cvx;
                        maxiter=maxiter_admm, disp_freq=disp_freq, wt=wt)
    comms_admm = 4 * ne(g) * ones(maxiter_admm)

    # Proximal Average
    maxiter_prox_ave = ceil(15000 / 2 / ne(g)) |> Int
    x_proxavg, obj_proxavg = proxavg(g, trainX, trainY, λ, initial_x, ss_proxavg,
                        gradient, pair_prox, loss, reg; maxiter=maxiter_prox_ave,
                        disp_freq=disp_freq)
    comms_proxavg = 2 * ne(g) * ones(maxiter_prox_ave)

    # Decentralized subgradient descent
    maxiter_dgd = ceil(15000 / 2 / ne(g)) |> Int
    x_dgd, obj_dgd = dgd(g, trainX, trainY, λ/2, initial_x_dgd, ss_dgd, grad_dgd,
            loss_dgd, reg; maxiter=maxiter_dgd, disp_freq=disp_freq)
    comms_dgd = 2 * ne(g) * ones(maxiter_dgd)

    # Walkman
    maxiter_walkman = ceil(10000 / num_nodes) |> Int
    # Note that we are indeed solving the original problem divided by num_nodes.
    x_walkman, obj_walkman = walkman(g, trainX, trainY, λ/num_nodes, initial_x_dgd,
            β_walkman, grad_walkman, loss_walkman, reg_walkman, prox_walkman;
            maxiter=maxiter_walkman, disp_freq=disp_freq)
    # To compare with other methods, we need to multiply the objective function
    # value by num_nodes.
    obj_walkman = obj_walkman .* num_nodes
    comms_walkman = num_nodes * ones(maxiter_walkman)

    if dispinfo
        println("\n\n")
        @printf "Convex.jl: %+10.7e\n" optval
        @printf "RandomEdge: %+10.7e\n" obj_block_prox[end]
        @printf "     ADMM: %+10.7e\n" obj_admm[end]
        @printf "  ProxAvg: %+10.7e\n" obj_proxavg[end]
        @printf "      DGD: %+10.7e\n" obj_dgd[end]
        @printf "  Walkman: %+10.7e\n" obj_walkman[end]
    end

    # Compute the objective value at the initial point, i.e., zeros.
    init_obj = [loss(trainX[i], trainY[i], zeros(dim +  1)) for i = 1:num_nodes] |> sum

    # Plot
    if showfig
        # Block proximal
        plot([0; cumsum(comms_block_prox)], [init_obj; obj_block_prox] .- optval, 
                dpi=300,
                label="RandomEdge", color=:black, linestyle=:solid, linewidth=1.5, 
                xlabel="Communication", ylabel=L"H(x) - H^*", yscale=:log10, 
                legend=:topright, tickfontsize=12, guidefontsize=16, titlefontsize=20,
                legendfontsize=14)
        # ADMM
        plot!([0; cumsum(comms_admm)], [init_obj; obj_admm] .- optval, label="ADMM", color=:red, linestyle=:dash, linewidth=1.5)
        # Proximal Average
        plot!([0; cumsum(comms_proxavg)], [init_obj; obj_proxavg] .- optval, label="proxavg", color=:blue, linestyle=:dashdot, linewidth=1.5)
        # Decentralized subgradient descent
        plot!([0; cumsum(comms_dgd)], [init_obj; obj_dgd] .- optval, label="DGD", color=:green, linestyle=:dashdotdot, linewidth=1.5)
        # Walkman
        plot!([0; cumsum(comms_walkman)], [init_obj; obj_walkman] .- optval, label="Walkman", color=:purple, linestyle=:auto, linewidth=1.5)
        title!(figtitle)
        xlims!(0, 10000) |> display
        if savefigure
            savefig(figname)
        end
    end

    # Save variables
    if savevar
        @save savename init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox, x_admm, obj_admm, comms_admm, x_proxavg, obj_proxavg, comms_proxavg, x_dgd, obj_dgd, comms_dgd, x_walkman, obj_walkman
    end

    return init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox,
           x_admm, obj_admm, comms_admm, x_proxavg, obj_proxavg, comms_proxavg,
           x_dgd, obj_dgd, comms_dgd, x_walkman, obj_walkman
end

"""
    init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox,
    x_admm, obj_admm, comms_admm, x_proxavg, obj_proxavg, comms_proxavg,
    x_dgd, obj_dgd, comms_dgd = run_syn_exp_LS_1norm(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node, λ,
        ss_blockprox, ss_dgd, ss_proxavg, β_walkman; same_prob=0.5, diff_prob=0.01,
        noise=1e-2, maxiter=10000, disp_freq=1, wt=ones(0, 0), loss=loss_LS,
        loss_dgd=loss_LS_dgd, loss_walkman=loss_LS_walkman, reg=reg1, 
        reg_walkman=reg1_walkman, loss_cvx=loss_LS_cvx,
        reg_cvx=reg1_cvx, gradient=grad_LS, grad_dgd=grad_LS_dgd,
        grad_walkman=grad_LS_walkman, pair_prox=pair_prox1, prox_walkman=prox1,
        gen_truex=gen_truex, gen_data=gen_data_LS, initial_x=initial_x,
        initial_x_dgd=initial_x_dgd, dispinfo=true,
        showfig=true, savefigure=false, savevar=false)

Run the synthetic experiment for the least squares problem with fused 1-norm regularization.

# Arguments
## Input:
    - `num_group::Int`: the number of groups in the graph.
    - `num_nodes_group::Vector{<:Int}`: the number of nodes in each group, a vector of size (num_group,).
    - `dim::Int`: the dimension of the data, the number of features.
    - `num_training_node::Vector{<:Int}`: the number of training nodes in each group,
        a vector of size (num_group,).
    - `num_testing_node::Vector{<:Int}`: the number of testing nodes in each group,
        a vector of size (num_group,).
    - `λ::Real`: the regularization parameter.
    - `ss_blockprox::Function`: the step size function for block proximal method.
    - `ss_dgd::Function`: the step size function for consensus optimization.
    - `ss_proxavg::Function`: the step size function for proximal average method.
    - `β_walkman::Real`: the step size for walkman.
    - `same_prob::Real`: the probability of having the same edge in the graph,
        default is 0.5.
    - `diff_prob::Real`: the probability of having a different edge in the graph,
        default is 0.01.
    - `noise::Real`: the noise level in the data generation, default is 1e-2.
    - `maxiter::Int`: the maximum number of iterations for the algorithms, default is 1000.
    - `disp_freq::Int`: the frequency of displaying the objective value, default is 1.
    - `wt::Matrix{<:Real}`: the weights of the edges in the graph, default is an
        empty matrix, which means the problem does not have weights.
    - `loss::Function`: a function to compute the local loss, it should take
        three arguments, the training data, the training label, and the variable,
        and return a real number.
            `lossval = loss(trainX[i], trainY[i], x[:, i])`
    - `loss_dgd::Function`: a function to compute the local loss of dgd, it should
        take seven arguments: the training data, the training label, the variable,
        the index of node, and the graph, λ, regularizer, and return a real number.
            `lossval = loss_dgd(X, Y, x, i, g, λ, reg)`
    - `loss_walkman::Function`: a function to compute the local loss of walkman,
        it should take four arguments: the training data, the training label,
        the variable, the index of node, and return a real number.
            `lossval = loss_walkman(X, Y, x, i)`
    - `reg::Function`: a function to compute the regularization term, it should
        take two arguments, the variable of the current node and the variable of
        the neighbor node, and return a real number.
            `regval = reg(x[:, i], x[:, neigh])`
    - `reg_walkman::Function`: a function to compute the whole regularizer, it should
        take three arguments, the variable of the current node, the graph, the
        regularization parameter, and return a real number.
            `regval = reg_walkman(x[:, i], g, λ)`
    - `loss_cvx::Function`: a function to compute the local loss for convex optimization,
        it should take three arguments, the training data, the training label, and
        the variable, and return a Convex.jl expression.
            `lossval = local_loss_cvx(trainX[i], trainY[i], x[:, i])`
    - `reg_cvx::Function`: a function to compute the regularization term for convex optimization,
        it should take two arguments, the variable of the current node and the variable
        of the neighbor node, and return a Convex.jl expression.
            `regval = reg_cvx(x[:, i], x[:, neigh])`
    - `gradient::Function`: a function to compute the gradient of the local loss,
        it should take three arguments, the training data, the training label, and
        the variable, and return a vector.
            `grad = gradient(trainX[i], trainY[i], x[:, i])`
    - `grad_dgd::Function`: a function to compute the gradient of the local loss for dgd,
        it should take seven arguments: the training data, the training label,
        the variable, the index of node, the graph, λ, regularizer, and return a vector.
            `grad = grad_dgd(X, Y, x, i, g, λ, reg)`
    - `grad_walkman::Function`: a function to compute the gradient of the local
        loss for walkman, it should take four arguments: the training data, the
        training label, the variable, the index of node, and return a vector.
            `grad = grad_walkman(X, Y, x, i)`
    - `pair_prox::Function`: a function to compute the proximal operator for the pair
        of variables, it should take three arguments, the variable of the current node,
        the variable of the neighbor node, and the step size, and return a vector.
            `y = pair_prox(x[:, i], x[:, neigh], ss, λ)`
    - `prox_walkman::Function`: a function to compute the proximal operator of
        the whole regulaizer, it should take three arguments, the variable,
        the graph, and the regulaization parameter.
            `y = prox_walkman(x[:, i], g, λ)`
    - `gen_truex::Function`: a function to generate the true variable, it should
        take two arguments, the number of groups and the dimension, and return a vector.
            `x_true = gen_truex(num_group, dim)`
    - `gen_data::Function`: a function to generate the training data and labels,
        it should take four arguments, the number of nodes, the dimension, the true variable,
        and the noise level, and return a tuple of training data and labels.
            `trainX, trainY = gen_data(num_nodes, dim, x_true, noise)`
    - `initial_x::Function`: a function to initialize the variable, it should take
        two arguments, the dimension and the number of nodes, and return a matrix.
            `x = initial_x(dim, num_nodes)`
    - `initial_x_dgd::Function`: a function to initialize the variable for dgd, it should take
        two arguments, the dimension and the number of nodes, and return a matrix.
            `x = initial_x_dgd(dim, num_nodes)`
    - `dispinfo::Bool`: whether to display the information during the experiment,
        default is true.
    - `showfig::Bool`: whether to show the figure of the results, default is true.
    - `savefigure::Bool`: whether to save the figure of the results, default is false.
    - `savevar::Bool`: whether to save the variables of the results, default is false.

## Output:
    - `init_obj<:Real`: the initial objective value, a vector of size (1,).
    - `x_cvx::Vector{<:Real}`: the optimal solution from convex optimization.
    - `optval::Real`: the optimal value from convex optimization.
    - `x_block_prox::Matrix{<:Real}`: the solution from block proximal method.
    - `obj_block_prox::Vector{<:Real}`: the objective values from block proximal method.
    - `comms_block_prox::Vector{<:Real}`: the number of communications from block proximal method.
    - `x_proxavg::Matrix{<:Real}`: the solution from proximal average method.
    - `obj_proxavg::Vector{<:Real}`: the objective values from proximal average method.
    - `comms_proxavg::Vector{<:Real}`: the number of communications from proximal average method.
    - `x_dgd::Matrix{<:Real}`: the solution from decentralized subgradient descent method.
    - `obj_dgd::Vector{<:Real}`: the objective values from decentralized subgradient descent method.
    - `comms_dgd::Vector{<:Real}`: the number of communications from decentralized subgradient descent method.
    - `x_walkman::Matrix{<:Real}`: the solution from walkman.
    - `obj_walkman::Vector{<:Real}`: the objective values from walkman, note that this one is already multiplied by num_nodes.
"""
function run_syn_exp_LS_1norm(
    num_group::Int,
    num_nodes_group::Vector{<:Int},
    dim::Int,
    num_training_node::Vector{<:Int},
    num_testing_node::Vector{<:Int},
    λ::Real,
    ss_blockprox::Function,
    ss_dgd::Function,
    ss_proxavg::Function,
    β_walkman::Real;
    same_prob::Real=0.5,
    diff_prob::Real=0.01,
    noise::Real=1e-2,
    maxiter::Int=10000,
    disp_freq::Int=1,
    wt::Matrix{<:Real}=ones(0, 0),
    loss::Function=loss_LS,
    loss_dgd::Function=loss_LS_dgd,
    loss_walkman::Function=loss_LS_walkman,
    reg::Function=reg1,
    reg_walkman::Function=reg1_walkman,
    loss_cvx::Function=loss_LS_cvx,
    reg_cvx::Function=reg1_cvx,
    gradient::Function=grad_LS,
    grad_dgd::Function=grad_LS_dgd,
    grad_walkman::Function=grad_LS_walkman,
    pair_prox::Function=pair_prox1,
    prox_walkman::Function=prox1,
    gen_truex::Function=gen_truex,
    gen_data::Function=gen_data_LS,
    initial_x::Function=initial_x,
    initial_x_dgd::Function=initial_x_dgd,
    dispinfo::Bool=true,
    showfig::Bool=true,
    savefigure::Bool=false,
    savevar::Bool=false,
)

    # Figure title.
    figtitle = "LS" * " with fused " * "1-norm regularization"
    # Figure name.
    figname = "LS_1_" * string(num_group) * "_" * Dates.format(now(), "yymmddHHMMSS") * ".png"
    # File name.
    savename = "LS_1_" * string(num_group) * "_" * Dates.format(now(), "yymmddHHMMSS") * ".jld2"

    # Generate the graph.
    g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges = gen_graph(
        num_group, num_nodes_group, dim, num_training_node, num_testing_node, gen_truex,
        gen_data; noise=noise, same_prob=same_prob, diff_prob=diff_prob)

    # We need to avoid unconnected graph since our competing methods require
    # connectivity.
    while !is_connected(g)
        g, x_true, trainX, trainY, testX, testY, groups, degrees, correct_edges = gen_graph(
            num_group, num_nodes_group, dim, num_training_node, num_testing_node, gen_truex,
            gen_data; noise=noise, same_prob=same_prob, diff_prob=diff_prob)
    end

    # Convex.jl
    x_cvx, optval = whole_cvx(g, trainX, trainY, λ, loss_cvx, reg_cvx; wt=wt)

    # Block Proximal
    x_block_prox, obj_block_prox, comms_block_prox = blockprox(g, trainX,
        trainY, λ, degrees, initial_x, ss_blockprox, gradient, pair_prox, loss,
        reg; maxiter=maxiter, disp_freq=disp_freq, wt=wt)

    # Proximal Average
    maxiter_prox_ave = ceil(15000 / 2 / ne(g)) |> Int
    x_proxavg, obj_proxavg = proxavg(g, trainX, trainY, λ, initial_x, ss_proxavg,
                        gradient, pair_prox, loss, reg; maxiter=maxiter_prox_ave,
                        disp_freq=disp_freq)
    comms_proxavg = 2 * ne(g) * ones(maxiter_prox_ave)

    # Decentralized subgradient descent
    maxiter_dgd = ceil(15000 / 2 / ne(g)) |> Int
    x_dgd, obj_dgd = dgd(g, trainX, trainY, λ, initial_x_dgd, ss_dgd, grad_dgd,
            loss_dgd, reg; maxiter=maxiter_dgd, disp_freq=disp_freq)
    comms_dgd = 2 * ne(g) * ones(maxiter_dgd)

    # Walkman
    maxiter_walkman = ceil(10000 / num_nodes) |> Int
    # Note that we are indeed solving the original problem divided by num_nodes.
    x_walkman, obj_walkman = walkman(g, trainX, trainY, λ/num_nodes, initial_x_dgd,
            β_walkman, grad_walkman, loss_walkman, reg_walkman, prox_walkman;
            maxiter=maxiter_walkman, disp_freq=disp_freq)
    # To compare with other methods, we need to multiply the objective function
    # value by num_nodes.
    obj_walkman = obj_walkman .* num_nodes
    comms_walkman = num_nodes * ones(maxiter_walkman)

    if dispinfo
        println("\n\n")
        @printf "Convex.jl: %+10.7e\n" optval
        @printf "RandomEdge: %+10.7e\n" obj_block_prox[end]
        @printf "     ADMM: %+10.7e\n" obj_admm[end]
        @printf "  ProxAvg: %+10.7e\n" obj_proxavg[end]
        @printf "      DGD: %+10.7e\n" obj_dgd[end]
        @printf "  Walkman: %+10.7e\n" obj_walkman[end]
    end

    # Compute the objective value at the initial point, i.e., zeros.
    init_obj = [loss(trainX[i], trainY[i], zeros(dim +  1)) for i = 1:num_nodes] |> sum

    # Plot
    if showfig
        # Block proximal
        plot([0; cumsum(comms_block_prox)], [init_obj; obj_block_prox] .- optval, 
                dpi=300,
                label="RandomEdge", color=:black, linestyle=:solid, linewidth=1.5, 
                xlabel="Communication", ylabel=L"H(x) - H^*", yscale=:log10, 
                legend=:topright, tickfontsize=12, guidefontsize=16, titlefontsize=20,
                legendfontsize=14)
        # Proximal Average
        plot!([0; cumsum(comms_proxavg)], [init_obj; obj_proxavg] .- optval, label="proxavg", color=:blue, linestyle=:dashdot, linewidth=1.5)
        # Decentralized subgradient descent
        plot!([0; cumsum(comms_dgd)], [init_obj; obj_dgd] .- optval, label="DGD", color=:green, linestyle=:dashdotdot, linewidth=1.5)
        # Walkman
        plot!([0; cumsum(comms_walkman)], [init_obj; obj_walkman] .- optval, label="Walkman", color=:purple, linestyle=:auto, linewidth=1.5)
        title!(figtitle)
        xlims!(0, 10000) |> display
        if savefigure
            savefig(figname)
        end
    end

    # Save variables
    if savevar
        @save savename init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox, x_proxavg, obj_proxavg, comms_proxavg, x_dgd, obj_dgd, comms_dgd, x_walkman, obj_walkman
    end

    return init_obj, x_cvx, optval, x_block_prox, obj_block_prox, comms_block_prox,
           x_proxavg, obj_proxavg, comms_proxavg, x_dgd, obj_dgd, comms_dgd,
           x_walkman, obj_walkman
end

"""
    run_multiple_exps_ls(num_group, num_nodes_group, dim, num_training_node,
        num_testing_node, λ, maxiter, num_exps; same_prob=0.5, diff_prob=0.01,
        noise=1e-2, disp_freq=15000, p=1/10, yscale=:log10, savefile=false,
        savefigure=false)

Run multiple experiments for the least squares problem with fused 1-norm and 2-norm regularization.

# Arguments
## Input:
    - `num_group::Int`: the number of groups in the graph.
    - `num_nodes_group::Vector{<:Int}`: the number of nodes in each group, a vector of size (num_group,).
    - `dim::Int`: the dimension of the data, the number of features.
    - `num_training_node::Vector{<:Int}`: the number of training nodes in each group,
        a vector of size (num_group,).
    - `num_testing_node::Vector{<:Int}`: the number of testing nodes in each group,
        a vector of size (num_group,).
    - `λ::Real`: the regularization parameter.
    - `maxiter::Int`: the maximum number of iterations for the algorithms.
    - `num_exps::Int`: the number of experiments to run.
    - `same_prob::Real`: the probability of having the same edge in the graph,
        default is 0.5.
    - `diff_prob::Real`: the probability of having a different edge in the graph,
        default is 0.01.
    - `noise::Real`: the noise level in the data generation, default is 1e-2.
    - `disp_freq::Int`: the frequency of displaying the objective value, default is 15000.
    - `yscale::Symbol`: scale for y-axis, default is :log10.
    - `savefile::Bool`: whether to save results to file, default is false.
    - `savefigure::Bool`: whether to save figures, default is false.
"""
function run_multiple_exps_ls(
    num_group::Int,
    num_nodes_group::Vector{<:Int},
    dim::Int,
    num_training_node::Vector{<:Int},
    num_testing_node::Vector{<:Int},
    λ::Real,
    maxiter::Int,
    num_exps::Int; 
    same_prob::Real=0.5,
    diff_prob::Real=0.01,
    noise::Real=1e-2,
    disp_freq::Int=15000,
    yscale=:log10,
    savefile::Bool=false,
    savefigure::Bool=false,
)
    
    # Number of nodes.
    num_nodes = sum(num_nodes_group)

    # Initialize storage for results
    initobj = zeros(Float64, 2, num_exps)  # [2-norm; 1-norm]
    x_cvx = zeros(Float64, 2, (dim + 1) * num_nodes, num_exps)
    optval = zeros(Float64, 2, num_exps)
    
    # Storage for BlockProx results
    x_blockprox = zeros(Float64, 2, dim+1, num_nodes, num_exps)
    obj_blockprox = zeros(Float64, 2, maxiter, num_exps)
    comms_blockprox = zeros(Float64, 2, maxiter, num_exps)
    error_blockprox = zero(obj_blockprox)
    
    # Storage for other algorithms
    x_admm = zeros(Float64, dim+1, num_nodes, num_exps)
    obj_admm = Vector{Vector{Float64}}(undef, num_exps)
    comms_admm = Vector{Vector{Int}}(undef, num_exps)
    error_admm = Vector{Vector{Float64}}(undef, num_exps)
    min_iter_admm = Inf
    
    x_proxavg = zeros(Float64, 2, dim+1, num_nodes, num_exps)
    obj_proxavg = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    comms_proxavg = [Vector{Vector{Int}}(undef, num_exps) for _ in 1:2]
    error_proxavg = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    min_iter_proxavg = [Inf, Inf]
    
    x_dgd = zeros(Float64, 2, (dim+1) * num_nodes, num_nodes, num_exps)
    obj_dgd = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    comms_dgd = [Vector{Vector{Int}}(undef, num_exps) for _ in 1:2]
    error_dgd = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    min_iter_dgd = [Inf, Inf]

    x_walkman = zeros(Float64, 2, (dim+1) * num_nodes, num_nodes, num_exps)
    obj_walkman = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    error_walkman = [Vector{Vector{Float64}}(undef, num_exps) for _ in 1:2]
    
    # Run experiments
    for i = 1:num_exps
        println("==== Experiment $i ====")

        # 1-norm experiments
        initobj[1,i], x_cvx[1,:,i], optval[1,i],
        x_blockprox[1,:,:,i], obj_blockprox[1,:,i], comms_blockprox[1,:,i],
        x_proxavg[1,:,:,i], obj_proxavg[1][i], comms_proxavg[1][i],
        x_dgd[1,:,:,i], obj_dgd[1][i], comms_dgd[1][i], x_walkman[1,:,:,i], 
        obj_walkman[1][i] = run_syn_exp_LS_1norm(num_group, num_nodes_group, dim,
            num_training_node, num_testing_node, λ, ss_ooqk, ss_const, ss_const,
            1e4; same_prob=same_prob, diff_prob=diff_prob, noise=noise,
            maxiter=maxiter, disp_freq=disp_freq, showfig=false, dispinfo=false)
                
        
        # 2-norm experiments
        initobj[2,i], x_cvx[2,:,i], optval[2,i], 
        x_blockprox[2,:,:,i], obj_blockprox[2,:,i], comms_blockprox[2,:,i],
        x_admm[:,:,i], obj_admm[i], comms_admm[i], 
        x_proxavg[2,:,:,i], obj_proxavg[2][i], comms_proxavg[2][i],
        x_dgd[2,:,:,i], obj_dgd[2][i], comms_dgd[2][i], x_walkman[2,:,:,i],
        obj_walkman[2][i] = run_syn_exp_LS_2norm(num_group, num_nodes_group, dim,
            num_training_node, num_testing_node, λ, ss_ooqk, ss_const, ss_const,
            1e4; same_prob=same_prob, diff_prob=diff_prob, noise=noise,
            maxiter=maxiter, disp_freq=disp_freq, showfig=false, dispinfo=false)
                
        # Calculate errors
        error_admm[i] = obj_admm[i] .- optval[2,i]
        # Compute the minimum iteration for ADMM.
        min_iter_admm = min(length(obj_admm[i]), min_iter_admm) |> Int
        
        for j in 1:2
            error_blockprox[j,:,i] = obj_blockprox[j,:,i] .- optval[j,i]
            error_proxavg[j][i] = obj_proxavg[j][i] .- optval[j,i]
            error_dgd[j][i] = obj_dgd[j][i] .- optval[j,i]
            error_walkman[j][i] = obj_walkman[j][i] .- optval[j,i]
            
            # Compute the minimum iteration for each method.
            min_iter_proxavg[j] = min(length(obj_proxavg[j][i]), min_iter_proxavg[j]) |> Int
            min_iter_dgd[j] = min(length(obj_dgd[j][i]), min_iter_dgd[j]) |> Int
        end
    end

    # Process results
    # Pick the first `min_iter` iterations for each method.
    # This is to ensure that we have the same number of iterations for each method
    # for plotting and analysis.
    for i = 1:num_exps
        error_admm[i] = error_admm[i][1:min_iter_admm]
        comms_admm[i] = comms_admm[i][1:min_iter_admm]
        
        for j in 1:2
            error_proxavg[j][i] = error_proxavg[j][i][1:Int(min_iter_proxavg[j])]
            error_dgd[j][i] = error_dgd[j][i][1:Int(min_iter_dgd[j])]
            
            comms_proxavg[j][i] = comms_proxavg[j][i][1:Int(min_iter_proxavg[j])]
            comms_dgd[j][i] = comms_dgd[j][i][1:Int(min_iter_dgd[j])]
        end
    end

    # Calculate mean communications
    mean_comms = Dict()
    mean_comms["blockprox"] = [mean(comms_blockprox[j,:,:], dims=2)[:] for j in 1:2]
    mean_comms["admm"] = mean(hcat(comms_admm...), dims=2)[:]
    mean_comms["proxavg"] = [mean(hcat(comms_proxavg[j]...), dims=2)[:] for j in 1:2]
    mean_comms["dgd"] = [mean(hcat(comms_dgd[j]...), dims=2)[:] for j in 1:2]
    mean_comms["walkman"] = [num_nodes * ones(ceil(10000 / num_nodes) |> Int) for _ in 1:2]

    # Create error matrices
    error_matrices = Dict()
    error_matrices["admm"] = hcat(error_admm...)
    for j in 1:2
        error_matrices["proxavg$j"] = hcat(error_proxavg[j]...)
        error_matrices["dgd$j"] = hcat(error_dgd[j]...)
        error_matrices["walkman$j"] = hcat(error_walkman[j]...)
    end

    # Save results if required
    if savefile
        filename = "LS_$(num_group)_$(num_nodes)_$(num_exps)" * ".jld2"
        JLD2.save(filename, Dict(
            "initobj" => initobj,
            "x_cvx" => x_cvx,
            "optval" => optval,
            "x_blockprox" => x_blockprox,
            "obj_blockprox" => obj_blockprox,
            "comms_blockprox" => comms_blockprox,
            "x_admm" => x_admm,
            "obj_admm" => obj_admm,
            "comms_admm" => comms_admm,
            "x_proxavg" => x_proxavg,
            "obj_proxavg" => obj_proxavg,
            "comms_proxavg" => comms_proxavg,
            "x_dgd" => x_dgd,
            "obj_dgd" => obj_dgd,
            "comms_dgd" => comms_dgd,
            "x_walkman" => x_walkman,
            "obj_walkman" => obj_walkman,
            "error_blockprox" => error_blockprox,
            "error_admm" => error_admm,
            "error_proxavg" => error_proxavg,
            "error_dgd" => error_dgd,
            "error_walkman" => error_walkman,
            "min_iter_admm" => min_iter_admm,
            "min_iter_proxavg" => min_iter_proxavg,
            "min_iter_dgd" => min_iter_dgd,
            "mean_comms" => mean_comms,
            "error_matrices" => error_matrices))
    end

    # Plot results
    for norm_type in 1:2
        figname = "LS_$(norm_type)_$(num_group)_$(num_nodes)_$(num_exps)" * ".png"

        # BlockProx
        # We first compute the error for each experiment and then take the mean and std.
        # Note that we use the std of the log10 of the errors to plot the ribbon.
        # This is because the errors can vary significantly across experiments,
        # causing the lower bound to be negative.
        error_blockprox_ = [(initobj[norm_type,:] .- optval[norm_type, :])';
             error_blockprox[norm_type,:,:]]
        gm_blockprox = mean(error_blockprox_, dims=2)[:]
        ls_blockprox = std(log10.(error_blockprox_); dims=2)[:]
        lower_blockprox = 10.0 .^(log10.(gm_blockprox) .- ls_blockprox)
        upper_blockprox = 10.0 .^(log10.(gm_blockprox) .+ ls_blockprox)
        
        plot([0; cumsum(mean_comms["blockprox"][norm_type])], gm_blockprox;
            yscale=yscale, ribbon = (gm_blockprox .- lower_blockprox, 
            upper_blockprox .- gm_blockprox), fillalpha=0.15, label="RandomEdge",
            dpi=300, xlabel="Communication", ylabel=L"H(x) - H^*", legend=:best,
            tickfontsize=12, guidefontsize=16, titlefontsize=20, legendfontsize=14,
            linestyle=:solid, linewidth=1.5, color=:blue)

        xlims!(0, 2maxiter)
        title!("LS with fused $(norm_type)-norm regularization")
        
        # Add other algorithms
        if norm_type == 2
            # ADMM
            error_admm_ = [(initobj[2,:] .- optval[2, :])'; error_matrices["admm"]]
            gm_admm = mean(error_admm_, dims=2)[:]
            ls_admm = std(log10.(error_admm_); dims=2)[:]
            lower_admm = 10.0 .^(log10.(gm_admm) .- ls_admm)
            upper_admm = 10.0 .^(log10.(gm_admm) .+ ls_admm)

            plot!([0; cumsum(mean_comms["admm"])], gm_admm; 
                ribbon = (gm_admm .- lower_admm, upper_admm .- gm_admm), 
                fillalpha=0.15, label="ADMM", marker=:utriangle, linewidth=1.5,
                color=:red)
        end
        
        for (i, alg) in enumerate(["proxavg", "dgd"])
            error__ = [(initobj[norm_type,:] .- optval[norm_type, :])';
                error_matrices["$(alg)$norm_type"]]
            gm_alg = mean(error__, dims=2)[:]
            ls_alg = std(log10.(error__); dims=2)[:]
            lower_alg = 10.0 .^(log10.(gm_alg) .- ls_alg)
            upper_alg = 10.0 .^(log10.(gm_alg) .+ ls_alg)
            plot!([0; cumsum(mean_comms["$(alg)"][norm_type])], gm_alg; 
                ribbon = (gm_alg .- lower_alg, upper_alg .- gm_alg), 
                fillalpha=0.15, label=["ProxAvg", "DSGD"][i],
                marker=[:xcross, :diamond][i],
                color=[:green, :purple][i], linewidth=1.5)
        end

        error__ = [(initobj[norm_type,:] .- optval[norm_type, :])';
            error_matrices["walkman$norm_type"]]
        gm_alg = mean(error__, dims=2)[:]
        ls_alg = std(log10.(error__); dims=2)[:]
        lower_alg = 10.0 .^(log10.(gm_alg) .- ls_alg)
        upper_alg = 10.0 .^(log10.(gm_alg) .+ ls_alg)
        plot!([0; cumsum(mean_comms["walkman"][norm_type])], gm_alg; 
            ribbon = (gm_alg .- lower_alg, upper_alg .- gm_alg), 
            fillalpha=0.15, label="Walkman",
            linestyle=:dash, color=:inferno, linewidth=1.5)
        
        display(current())

        # Save figure if required
        if savefigure
            savefig(figname)
        end
    end
end