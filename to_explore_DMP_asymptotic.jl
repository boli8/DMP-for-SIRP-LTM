"""
Main programs for exploring the macroscopic behaviors of the SIRP+LTM model in the asymptotic limit.
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random
using Statistics
using ReverseDiff
using Plots

include("GraphUtil.jl")
# using .GraphUtil
include("DMP_funcs.jl")
include("Optim_funcs.jl")


## Directory to dump results:
dir_result = "./results/"

## Directory to load networks:
dir_network = "./networks/"

## Specify the networks:
graph_name_a = "rrg_N100_d5_seed100"
graph_name_b = "rrg_N100_d5_seed200"

# graph_name_a = "rrg_N1600_d5_seed100"
# graph_name_b = "rrg_N1600_d5_seed200"


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
T = 100
μmag = 0.5
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

βmags = collect(0.05 : 0.05 : 0.5)
dat = zeros(length(βmags), 3)  ## for recording the numerical results

for (idx, βmag) in enumerate(βmags)
    β = adj_mat_a * βmag
    βv = ones(no_of_edges_a) * βmag
  
    ## Marginal probabilities by DMP:                                        
    PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = 
        forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, 
                                σ0, βv, γ, μ, adj_n_b, deg_b, b, θ, 
                                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

    ## Expected number of individuals affected:
    ρS_dmp = mean(PS_mgn, dims=2)[:, 1] 
    ρI_dmp = mean(PI_mgn, dims=2)[:, 1]
    ρR_dmp = mean(PR_mgn, dims=2)[:, 1]
    ρP_dmp = mean(PP_mgn, dims=2)[:, 1]
    ρF_dmp = mean(PF_mgn, dims=2)[:, 1]

    ## Record ρI[T] + ρR[T] and ρF[T]:
    dat[idx, :] = [βmag, ρI_dmp[T] + ρR_dmp[T], ρF_dmp[T]]

    ## set "save_soln = true" to save solutions:
    save_soln = true   
    if save_soln
        df = DataFrame(t = collect(0:T), rho_S_dmp = ρS_dmp, rho_I_dmp = ρI_dmp, rho_R_dmp = ρR_dmp, 
                       rho_P_dmp = ρP_dmp, rho_F_dmp = ρF_dmp)
        CSV.write(dir_result * "macro_stat_DMP_GA_" * graph_name_a * "_GB_" * graph_name_b * "_beta$(βmag).csv",
                  df, writeheader=true)
    end
end

plot(dat[:, 1], dat[:, 2], xlabel="beta", ylabel="rho", label="rho_I(T) + rho_R(T)", marker=:circle)
plot!(dat[:, 1], dat[:, 3], xlabel="beta", ylabel="rho", label="rho_F(T)", marker=:circle)
savefig(dir_result * "exploring_DMP_asymptotic.png")
