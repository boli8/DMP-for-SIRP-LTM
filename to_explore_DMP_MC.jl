"""
Main programs for exploring dynamic message-passing of the SIRP+LTM model,
and compare it to the Monte Carlo (MC) simulations.
"""

using DataFrames
using CSV
using LightGraphs
using LinearAlgebra
using Random
using Statistics
using SparseArrays
using Plots

include("GraphUtil.jl")
# using .GraphUtil
include("DMP_funcs.jl")
include("MC_funcs.jl")


## Directory to dump results:
dir_result = "./results/"

## Directory to load networks:
dir_network = "./networks/"

## Specify the networks:
# graph_name_a = "binary_tree_depth6"
# graph_name_b = "binary_tree_depth6"

graph_name_a = "rrg_N100_d5_seed100"
graph_name_b = "rrg_N100_d5_seed100"


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

## randomly select some nodes to be initially infected (i.e., seeds)
no_of_seeds = 5
Random.seed!(101)
ord = randperm(no_of_nodes_a)
σ0 = zeros(Float64, no_of_nodes_a)
σ0[ ord[1:no_of_seeds] ] .= 1

opt = "gamma"
γ = zeros(T, no_of_nodes_a)
# γ = rand(T, no_of_nodes_a) / 10

## Marginal probabilities by DMP:                                        
PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = 
    forward_all_states_SIRP_LTM_approx(T, edge_list_a, adj_n_a, deg_a, 
                            σ0, βv, γ, μ, adj_n_b, deg_b, b, θ, 
                            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

## Marginal probabilities by MC:
PS_mc, PI_mc, PR_mc, PP_mc, PF_mc = SIRP_LTM_MC_sim(T, adj_mat_a, adj_n_a, deg_a, σ0, β, γ, μ,
                                        adj_mat_b, adj_n_b, deg_b, b, θ)

## Outbreak sizes:
Ts = collect(0:T)
ρIR_dmp = mean(PI_mgn, dims=2)[:, 1] + mean(PR_mgn, dims=2)[:, 1]
ρF_dmp  = mean(PF_mgn, dims=2)[:, 1]
ρIR_mc  = mean(PI_mc, dims=2)[:, 1] + mean(PR_mc, dims=2)[:, 1]
ρF_mc   = mean(PF_mc, dims=2)[:, 1]

## Compare MC and DMP:
p1 = plot(Ts, [ρIR_dmp ρF_dmp], xlabel="t", ylabel="rho", label=["rho_IR_dmp" "rho_F_dmp"])
p2 = plot(Ts, [ρIR_mc  ρF_mc],  xlabel="t", ylabel="rho", label=["rho_IR_mc"  "rho_F_mc"])
p3 = plot(vec(PS_mc), vec(PS_mgn), seriestype=:scatter, xlabel="PS_mc[t, i]", ylabel="PS_dmp[t, i]", aspect_ratio=1)
p4 = plot(vec(PF_mc), vec(PF_mgn), seriestype=:scatter, xlabel="PF_mc[t, i]", ylabel="PF_dmp[t, i]", aspect_ratio=1)
plot!(p3, [0, 1], [0, 1], xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], label="identity")
plot!(p4, [0, 1], [0, 1], xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], label="identity")
plot(p1, p2, p3, p4, layout=(2, 2))
savefig(dir_result * "exploring_DMP_MC.png")


## set "save_mc_soln" and/or "save_dmp_soln" as "true" save solutions:
save_dmp_soln = true   
save_mc_soln  = true  

if save_dmp_soln
    prefix = "states_dmp_"
    graph_names = "_GA_" * graph_name_a * "_GB_" * graph_name_b
    CSV.write(dir_result * prefix * "PS" * graph_names * ".csv", DataFrame(PS_mgn, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PI" * graph_names * ".csv", DataFrame(PI_mgn, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PR" * graph_names * ".csv", DataFrame(PR_mgn, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PP" * graph_names * ".csv", DataFrame(PP_mgn, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PF" * graph_names * ".csv", DataFrame(PF_mgn, :auto), writeheader=false)
end

if save_mc_soln
    prefix = "states_mc_"
    graph_names = "_GA_" * graph_name_a * "_GB_" * graph_name_b
    CSV.write(dir_result * prefix * "PS" * graph_names * ".csv", DataFrame(PS_mc, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PI" * graph_names * ".csv", DataFrame(PI_mc, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PR" * graph_names * ".csv", DataFrame(PR_mc, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PP" * graph_names * ".csv", DataFrame(PP_mc, :auto), writeheader=false)
    CSV.write(dir_result * prefix * "PF" * graph_names * ".csv", DataFrame(PF_mc, :auto), writeheader=false)
end
