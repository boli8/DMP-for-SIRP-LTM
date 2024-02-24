"""
Main programs for exploring the failure mitigation of the SIRP+LTM model,
by deploying protection resource (γ[t, i]).

The protection resources are subject to the constraint: sum_[t, i] γ[t, i] <= γtot.
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


## Load the Tailor shop net, and define the system paramters (as global variables):
include("to_load_Tailor-shop-net.jl")

## Load the N=118 synthetic net, and define the system paramters (as global variables):
## -- By default, planted b[j, i] are used.
# include("to_load_synthetic-net-N118.jl")



## ---- Begin case, No control ---- ##
γ = zeros(T, no_of_nodes_a)
objf_no_ctr = forward_obj_func_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = 
    forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, 
                            σ0, βv, γ, μ, adj_n_b, deg_b, b, θ, 
                            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

ρF_no_ctr = mean(PF_mgn, dims=2)[:, 1]
## ---- End case, No control ---- ##



## ---- Begin case, Random gamma control ---- ##
seed_gammas = collect(11:30)
ρF_rand_gamma_all = zeros(T+1, length(seed_gammas))

for (idx, seed_gamma) in enumerate(seed_gammas)
    Random.seed!(seed_gamma)
    ord_gamma = randperm(no_of_nodes_a)
    γ = zeros(T, no_of_nodes_a)
    γ[ 1, ord_gamma[1:γtot] ] .= 1

    PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, 
                        σ0, βv, γ, μ, adj_n_b, deg_b, b, θ, 
                        adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

    ρF_rand_gamma_all[:, idx] = mean(PF_mgn, dims=2)[:, 1]
end

ρF_rand_gamma_mean = mean(ρF_rand_gamma_all, dims=2)[:, 1]
ρF_rand_gamma_std  =  std(ρF_rand_gamma_all, dims=2)[:, 1]
## ---- End case, Random gamma control ---- ##



## ---- Begin case, Control SIRP only ---- ##
objf_func = forward_obj_func_SIRP

γ = zeros(T, no_of_nodes_a)
o1, o2, γ_soln_ctr_a, opt_log = 
    gradient_descent_over_γ_mirror_SIRP_LTM(objf_func, T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, γtot, "gamma")

## Optionally, round up γ[t, i] to be integer:
require_integer_γ = false
if require_integer_γ
    γ_soln_ctr_a = round_up_γ(T, no_of_nodes_a, γ_soln_ctr_a, γtot)
end

PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = 
    forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, 
                            σ0, βv, γ_soln_ctr_a, μ, adj_n_b, deg_b, b, θ, 
                            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

ρF_ctr_a = mean(PF_mgn, dims=2)[:, 1]

## To save solutions:
save_gamma = true
if save_gamma
    open(dir_result * "gamma_soln_ctr_a_at_gammatot$(γtot)" * "_GA_" * graph_name_a * "_GB_" * graph_name_b * ".csv", "w") do io
        write(io, "i,sigma0,")
        for t in 1:T
            write(io, "t$(t),")
        end
        write(io, "\n")

        for i in 1:no_of_nodes_a
            write(io, "$(i),$(σ0[i]),")
            for t in 1:T
                write(io, "$(γ_soln_ctr_a[t, i]),")
            end
            write(io, "\n")
        end
    end
end
## ---- End case, Control SIRP only ---- ##



## ---- Begin case, Control SIRP + LTM ---- ##
objf_func = forward_obj_func_SIRP_LTM_approx
# objf_func = forward_obj_func_SIRP_LTM   ## use the full expression of the LTM layer, which can be slow.

γ = zeros(T, no_of_nodes_a)
o1, o2, γ_soln_ctr_ab, opt_log = 
    gradient_descent_over_γ_mirror_SIRP_LTM(objf_func, T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, γtot, "gamma")

## Optionally, round up γ[t, i] to be integer:
require_integer_γ = false
if require_integer_γ
    γ_soln_ctr_ab = round_up_γ(T, no_of_nodes_a, γ_soln_ctr_ab, γtot)
end

PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn = 
    forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, 
                            σ0, βv, γ_soln_ctr_ab, μ, adj_n_b, deg_b, b, θ, 
                            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

ρF_ctr_ab = mean(PF_mgn, dims=2)[:, 1]

## To save solutions:
save_gamma = true
if save_gamma
    open(dir_result * "gamma_soln_ctr_ab_at_gammatot$(γtot)" * "_GA_" * graph_name_a * "_GB_" * graph_name_b * ".csv", "w") do io
        write(io, "i,sigma0,")
        for t in 1:T
            write(io, "t$(t),")
        end
        write(io, "\n")

        for i in 1:no_of_nodes_a
            write(io, "$(i),$(σ0[i]),")
            for t in 1:T
                write(io, "$(γ_soln_ctr_ab[t, i]),")
            end
            write(io, "\n")
        end
    end
end
## ---- End case, Control SIRP + LTM ---- ##



## Print and plot results
println("Outbreaksize at T, no control:     $(ρF_no_ctr[T])")
println("Outbreaksize at T, random control: $(ρF_rand_gamma_mean[T])")
println("Outbreaksize at T, control a:      $(ρF_ctr_a[T])")
println("Outbreaksize at T, control ab:     $(ρF_ctr_ab[T])")

Ts = collect(0:T)
plot(Ts, [ρF_no_ctr ρF_ctr_a ρF_ctr_ab], xlabel="t", ylabel="rho_F", legend=:bottomright,
     label=["no protection"  "minimizing rho_I(T) + rho_R(T)"  "minimizing rho_F(T)"])
plot!(Ts, ρF_rand_gamma_mean, yerr=ρF_rand_gamma_std, label="random gamma")
savefig(dir_result * "exploring_mitigation.png")
