# include("./macros.jl")
include("./src.jl")

# Set random seed.
Random.seed!(2)

# The number experiments.
num_exps = 100
# The yscale when plotting figures.
yscale = :log10

println("\n\n")

#===============================================================================
    Least squares problem.
    Multiple groups of nodes with different number of nodes.
===============================================================================#
# The number of groups.
num_group = 5;
# The number of nodes in each group.
num_nodes_group = [10, 17, 18, 18, 12] # rand(10:20, num_group); # 10 * ones(Int64, num_group);
# The number of nodes.
num_nodes = sum(num_nodes_group);
# The probability of connecting two nodes in the same group.
same_prob = 0.5;
# The probability of connecting two nodes in different groups.
diff_prob = 0.01;
# The dimension of data.
dim = 20;
# The number of training data points in each node.
num_training_node = 15 * ones(Int64, num_nodes);
# The number of testing data points in each node.
num_testing_node = 10 * ones(Int64, num_nodes);

λ = 1.0
maxiter = 5000
disp_freq = 15000
noise = 1e-2

# Run experiments of LS.
run_multiple_exps_ls(num_group, num_nodes_group, dim, num_training_node, num_testing_node, λ, maxiter, num_exps; 
      same_prob=same_prob, diff_prob=diff_prob, noise=noise, disp_freq=disp_freq, yscale=yscale, savefile=true,
      savefigure=true)

#===============================================================================
    Least squares problem.
    Single groups of nodes.
===============================================================================#

# The number of groups.
num_group = 1;
# The number of nodes in each group.
num_nodes_group = 20 * ones(Int64, num_group);
# The number of nodes.
num_nodes = sum(num_nodes_group);
# The probability of connecting two nodes in the same group.
same_prob = 0.5;
# The probability of connecting two nodes in different groups.
diff_prob = 0.01;
# The dimension of data.
dim = 20;
# The number of training data points in each node.
num_training_node = 15 * ones(Int64, num_nodes);
# The number of testing data points in each node.
num_testing_node = 10 * ones(Int64, num_nodes);

# The regularization parameter.
λ = 1.0
maxiter = 5000
disp_freq = 15000
noise = 1e-2

# Run experiments of LS.
run_multiple_exps_ls(num_group, num_nodes_group, dim, num_training_node, num_testing_node, λ, maxiter, num_exps; 
      same_prob=same_prob, diff_prob=diff_prob, noise=noise, disp_freq=disp_freq, yscale=yscale, savefile=true,
      savefigure=true)

#===============================================================================
    Least squares problem.
    Single groups of nodes, fully connected.
===============================================================================#

# The number of groups.
num_group = 1;
# The number of nodes in each group.
num_nodes_group = 40 * ones(Int64, num_group);
# The number of nodes.
num_nodes = sum(num_nodes_group);
# The probability of connecting two nodes in the same group.
same_prob = 1.0;
# The probability of connecting two nodes in different groups.
diff_prob = 1.0;
# The dimension of data.
dim = 20;
# The number of training data points in each node.
num_training_node = 15 * ones(Int64, num_nodes);
# The number of testing data points in each node.
num_testing_node = 10 * ones(Int64, num_nodes);

# The regularization parameter.
λ = 1.0
maxiter = 5000
disp_freq = 15000
noise = 1e-2

# Run experiments of LS.
run_multiple_exps_ls(num_group, num_nodes_group, dim, num_training_node, num_testing_node, λ, maxiter, num_exps; 
      same_prob=same_prob, diff_prob=diff_prob, noise=noise, disp_freq=disp_freq, yscale=yscale, savefile=true,
      savefigure=true)