# Decentralized Optimization with Topology-Independent Communication

This repository contains the Julia implementation of the `RandomEdge` algorithm, the specification of `BlockProx` to graph-guided regularizers.

`BlockProx` is a decentralized algorithm for solving the following structured optimization problem in a decentralized manner:

$$\min_{x \in \mathbb{R}^{nd}} \{H(x) := F(x) + G(x)\},$$

where the objective exhibits two distinct forms of separability:

$$F(x) := \sum_{i=1}^n f_i(x_i) \quad \text{and} \quad G(x) := \sum_{j=1}^m G_j(x).$$

The variable $x=(x_1, \ldots, x_n) \in \mathbb{R}^{nd}$ partions into blocks $x_i \in \mathbb{R}^d$, where each $f_i : \mathbb{R}^d \to \mathbb{R}$ and $G_j : \mathbb{R}^{nd} \to \mathbb{R}$ is proper, closed, and convex.
Each block $x_i$ corresponds to node $i$ in the communication network, where $f_i$ represents the local data and objective for node $i$, and $G_j$ represents the $j$ th coordination constraint.

For further details on the problem and the algorithm, please refer to [our paper](https://arxiv.org/abs/2509.14488):

```bibtex
@article{LKAF25BlockProx,
      title={Decentralized Optimization with Topology-Independent Communication}, 
      author={Ying Lin and Yao Kuang and Ahmet Alacaoglu and Michael P. Friedlander},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2509.14488}, 
}
```

## License

`BlockProx` is licensed under the MIT License (see [LICENSE](https://github.com/MPF-Optimization-Laboratory/BlockProx/blob/master/LICENSE)).

## Installation

1. Install Julia with version higher than 1.11, see [installation](https://julialang.org/install/).

2. Clone this repository.

   ```bash
   git clone git@github.com:MPF-Optimization-Laboratory/BlockProx.git
   cd BlockProx
   ```

3. Instantiate this project in `Julia` REPL:

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
   and wait for automatically installing required packages.

4. Run runcodes from command line:

   ```sh
   julia ./runcode_LS.jl
   julia ./runcode_housing.jl
   ```
   or inside `Julia` REPL:
   ```julia
   include("./runcode_LS.jl")
   include("./runcode_housing.jl")
   ```

## Code structure

- All the implementations are included in `src.jl`, with built-in detailed documentations and comments.
- There are two runcode files:
  1. `runcode_LS.jl`: The runcode for experiments on synthetic data: least squares benchmarks, see Section 6.1 of the paper.
  2. `runcode_housing.jl`: The runcode for experiments on real data: housing dataset, see Section 6.2 of the paper.
