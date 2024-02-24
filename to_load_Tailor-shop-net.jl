"""
Load the Tailor shop net, and define the system paramters.
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
graph_name_a = "Tailor_shop_a"
graph_name_b = "Tailor_shop_b"


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


## System parameters:
T = 10
βmag = 0.07 
μmag = 0.5
β = adj_mat_a * βmag
βv = ones(no_of_edges_a) * βmag
μ = ones(Float64, no_of_nodes_a) * μmag

bmag = 1.
θmag = 0.6
b = adj_mat_b * bmag
θ = deg_b * θmag

## At time 0, nodes with highest degrees in net a are infected (seeded)
ord = sortperm(deg_a, rev=true)
## top 7 highest-degree nodes in net a: [16, 19, 11, 12, 32, 34, 3]

no_of_seeds = 5
σ0 = zeros(Float64, no_of_nodes_a)
σ0[ ord[1:no_of_seeds] ] .= 1

opt = "gamma"
γtot = 10
γtot_each_t = 2     ## for online resource only
