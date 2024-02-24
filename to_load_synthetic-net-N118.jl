"""
Load a synthetic two-layer net (N=118), and define the system paramters.

- Layer a is a Watts-Strogatz small-world network, which mimics the topology of communication networks; 
it is obtained by rewiring a regular graph of degree 4 with rewiring probability p_rewire = 0.3. 
- Layer b is a power network extracted from the IEEE 118-bus test case.

Planted influence parameters b[j, i] in the LTM layer are considered.
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random
using Statistics

include("GraphUtil.jl")
# using .GraphUtil
include("DMP_funcs.jl")


## Directory to load networks:
dir_network = "./networks/"

## Specify the networks:
# graph_name_a = "binary_tree_depth6"
# graph_name_b = "binary_tree_depth6"

graph_name_a = "sw_n118_k4_p03_seed100"
graph_name_b = "ieee118"


## ---- Begin loading networks ---- ##
## net a:
node_data_a, edge_data_a, graph_a = GraphUtil.read_graph_from_csv(dir_network * graph_name_a, false)   # an un-directed graph
no_of_nodes_a = nv(graph_a)
no_of_edges_a = ne(graph_a)
max_deg_a, deg_a, edge_list_a, edge_indx_a, adj_n_a, adj_e_a, adj_e_indx_a, adj_mat_a, B_a = GraphUtil.undigraph_repr(graph_a, edge_data_a)

## net b:
node_data_b, edge_data_b, graph_b = GraphUtil.read_graph_from_csv(dir_network * graph_name_b, false)   # an un-directed graph
no_of_nodes_b = nv(graph_b)
no_of_edges_b = ne(graph_b)
max_deg_b, deg_b, edge_list_b, edge_indx_b, adj_n_b, adj_e_b, adj_e_indx_b, adj_mat_b, B_b = GraphUtil.undigraph_repr(graph_b, edge_data_b)

## Graph representation for the combined multiplex net:
adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb = build_two_layer_net(adj_n_a, deg_a, adj_n_b, deg_b)
## ---- End loading networks ---- ##


## Set planted influence parameters b[j, i]:
## -- Otherwise b[j, i] remains homogeneous.
set_planted_b = true

if set_planted_b
    ## System parameters:
    T = 50
    βmag = 0.17
    μmag = 0.5
    β = adj_mat_a * βmag
    βv = ones(no_of_edges_a) * βmag
    μ = ones(Float64, no_of_nodes_a) * μmag

    bmag = 1.
    θmag = 0.6
    b = adj_mat_b * bmag
    θ = deg_b * θmag

    ## Select some nodes to be initially infected (i.e., seeds)
    no_of_seeds = 3
    seeds = [36, 53, 6]
    σ0 = zeros(Float64, no_of_nodes_a)
    σ0[ seeds ] .= 1

    γ = zeros(T, no_of_nodes_a)
    γtot = 9
    γtot_each_t = 2     ## for online resource only; otherwise it is irrelevant
    opt = "gamma"

    ## A planted vulnerable path:
    path_vnrb = 
        [34, 43, 44, 45, 46, 47, 69, 77, 76, 118, 75, 70, 71, 72, 24, 23, 22, 21, 20, 19, 
        18, 17, 31, 32, 114, 115, 27, 25, 26, 30, 38, 65, 68, 81, 80, 97, 96, 82, 83, 84,
        85, 88, 89, 90, 91, 92, 102, 101, 100, 103, 104, 105, 108, 109, 110, 112]

    for m in 1:length(path_vnrb)-1
        k, i = path_vnrb[m], path_vnrb[m+1]
        b[k, i] = θ[i] + 0.01
    end

else
    ## System parameters:
    T = 50
    βmag = 0.2
    μmag = 0.5
    β = adj_mat_a * βmag
    βv = ones(no_of_edges_a) * βmag
    μ = ones(Float64, no_of_nodes_a) * μmag

    bmag = 1.
    θmag = 0.6
    b = adj_mat_b * bmag
    θ = deg_b * θmag

    ## Select some nodes to be initially infected (i.e., seeds)
    no_of_seeds = 5
    seeds = [36, 53, 6, 97, 49]
    σ0 = zeros(Float64, no_of_nodes_a)
    σ0[ seeds ] .= 1

    γ = zeros(T, no_of_nodes_a)
    γtot = 10
    γtot_each_t = 2     ## for online resource only; otherwise it is irrelevant
    opt = "gamma"

end
