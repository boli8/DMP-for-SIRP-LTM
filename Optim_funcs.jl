"""
Functions for optimizing the SIRP+LTM model.
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random
using ReverseDiff

include("DMP_funcs.jl")


"""
Objective function governed by the forward equations of SIRP only.
"""
function forward_obj_func_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)
    PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav = cal_dynamic_messages_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, opt)
    PS_mgn, PI_mgn, PR_mgn, PP_mgn = cal_DMP_marginal_SIRP(T, adj_n_a, deg_a, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)

    obj = sum( PI_mgn[T+1, :] ) + sum( PR_mgn[T+1, :] )
    return obj
end


"""
Objective function governed by the forward equations of SIRP+LTM.
Full expression for the LTM layer.
"""
function forward_obj_func_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)
    PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav = cal_dynamic_messages_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, opt)
    PS_mgn, PI_mgn, PR_mgn, PP_mgn = cal_DMP_marginal_SIRP(T, adj_n_a, deg_a, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)

    ψ_cav, PF_cav, χ_cav, χm_cav = cal_dynamic_messages_LTM_SIRP_full(T, edge_list_a, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
                PS_cav, PI_cav, PR_cav, θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, opt)
    PF_mgn = cal_DMP_marginal_LTM_SIRP_full(T, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
                θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, ψ_cav, PF_cav, χ_cav, χm_cav, opt)

    obj = sum( PF_mgn[T+1, :] )
    return obj
end


"""
Objective function governed by the forward equations of SIRP+LTM. 
Approximated expression for the LTM layer, assuming decorrelation of networks a and b.
"""
function forward_obj_func_SIRP_LTM_approx(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)
    PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav = cal_dynamic_messages_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, opt)
    PS_mgn, PI_mgn, PR_mgn, PP_mgn = cal_DMP_marginal_SIRP(T, adj_n_a, deg_a, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)

    σ0_b = PI_mgn[1, :] + PR_mgn[1, :]
    PF_cav = cal_dynamic_messages_LTM_by_PS_mgn(T, adj_mat_b, adj_n_b, deg_b, σ0_b, b, θ, PS_mgn+PP_mgn, opt)
    PF_mgn = cal_DMP_marginal_LTM_by_PS_mgn(T, adj_mat_b, adj_n_b, deg_b, σ0_b, b, θ, PS_mgn+PP_mgn, PF_cav, opt)

    obj = sum( PF_mgn[T+1, :] )
    return obj
end


"""
Mirror descent over γ, assuming sum_[t, i] γ[t, i] <= γtot.
----
In this experiment, a reparameterization method is use to enforce the constraint 0 < γ[t, i] < 1
    γ[t, i] = sigmoid(h[t, i]).
"""
function gradient_descent_over_γ_mirror_SIRP_LTM(objf_func, T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, γtot=10, opt="gamma"; seed=100, save_log=false)
    ## Setting the parameters:
    no_of_nodes_a = size(deg_a, 1)

    Random.seed!(seed)
    γ = rand(T, no_of_nodes_a)
    γ *= 0.05 * γtot / sum(γ)
    h = inv_sigmoid.(γ)
    
    nsteps = 30
    opt_log = zeros(nsteps+1, 3)

    o1 = objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

    ## objective function L(γ)
    L_of_γ = x -> objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, x, μ, adj_n_b, deg_b, b, θ,
                    adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, "gamma")

    ## Perform mirror descent:
    for step in 1:nsteps
        ∇L = ReverseDiff.gradient(L_of_γ, γ)
        h = inv_sigmoid.(γ)

        ## Subtracting the component of total γ increment if necessary:
        sum_γ = sum(γ)
        if sum_γ > 0.98 * γtot
            dγ = γ .* (1 .- γ)
            ∇L = ∇L .- sum(dγ .*  ∇L) / sum(dγ)
        end

        ## Backtracking line search:
        α, r = 0.3, 0.6     ## parameters for line search
        s = 20.             ## initial guess of step size
        L = L_of_γ(γ)
        L_temp = 0.

        for inner_step in 1:15
            γtemp = sigmoid.( h - s*∇L )
            γtot_temp = sum( γtemp )
            if γtot_temp > γtot    ## To ensure sum_[t, i] γ[t, i] < γtot
                s *= r
                continue
            end

            L_temp = L_of_γ(γtemp)
            if L_temp < L - α* sum(∇L .* (γtemp - γ))
                # L = L_temp
                break
            else
                s *= r
            end
        end

        γ = sigmoid.( h - s*∇L )
        # if sum(γ) > γtot
        #     γ *= 0.99 * γtot / sum(γ)
        #     γ = min.(max.(γ, 0), 1)
        # end
        γ *= 0.99 * γtot / sum(γ)
        γ = min.(max.(γ, 0), 1)
        
        sum_γ = sum(γ)
        ∇L_norm = sum(∇L .^ 2)
        println("At step = $(step), s = $(s), min_γ = $(minimum(γ)), max_γ = $(maximum(γ)), sum_γ = $(sum_γ), |∇L|^2 = $(∇L_norm), L = $(L_temp).")

        ## Saving logs:
        if save_log
            opt_log[step, :] = [L, sum_γ, ∇L_norm]
        end

        if ∇L_norm < 1e-6
            break
        end
    end

    o2 = objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

    println("\n")
    println("minimum and maximum of γ: $(minimum(γ)), $(maximum(γ))\n")
    println("sum γ before and after optimization: $(γtot), $(sum(γ))\n")
    println("obj before and after optimization: $(o1), $(o2)\n")

    return o1, o2, γ, opt_log
end


"""
Mirror descent over γ, assuming sum_[i] γ[t, i] <= γtot_each_t. 
Resources are deployed in an online manner.
----
In this experiment, a reparameterization method is use to enforce the constraint 0 < γ[t, i] < 1
    γ[t, i] = sigmoid(h[t, i]).
"""
function gradient_descent_over_γ_mirror_SIRP_LTM_online_resource(objf_func, T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, γtot_each_t=1, opt="gamma"; seed=300, save_log=false)
    ## Setting the parameters:
    no_of_nodes_a = size(deg_a, 1)

    Random.seed!(seed)
    γ = rand(T, no_of_nodes_a)
    h = zeros(T, no_of_nodes_a)
    for t in 1:T
        γ[t, :] *= 0.05 * γtot_each_t / sum(γ[t, :])
        h[t, :] = inv_sigmoid.(γ[t, :])
    end

    nsteps = 100
    opt_log = zeros(nsteps+1, 3)

    o1 = objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)


    ## objective function L(γ)
    L_of_γ = x -> objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, x, μ, adj_n_b, deg_b, b, θ,
                    adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, "gamma")

    ## Perform gradient ascent:
    for step in 1:nsteps
        ∇L = ReverseDiff.gradient(L_of_γ, γ)
        h = inv_sigmoid.(γ)

        ## Subtracting the component of total γ_t increment:
        dγ = γ .* (1 .- γ)
        for t in 1:T
            if sum(γ[t, :]) > 0.98 * γtot_each_t
                ∇L[t, :] = ∇L[t, :] .- sum(dγ[t, :] .*  ∇L[t, :]) / sum(dγ[t, :])
            end
        end

        ## Backtracking line search:
        α, r = 0.3, 0.6     ## parameters for line search
        s = 20.             ## initial guess of step size
        L = L_of_γ(γ)
        L_temp = 0.

        for inner_step in 1:15
            γtemp = sigmoid.( h - s*∇L )
            γtot_each_t_temp = sum( γtemp, dims=2 )
            if sum( γtot_each_t_temp .> γtot_each_t ) > 0    ## To ensure sum_[i] γ[t, i] < γtot_each_t
                s *= r
                continue
            end

            L_temp = L_of_γ(γtemp)
            if L_temp < L - α* sum(∇L .* (γtemp - γ))
                L = L_temp
                break
            else
                s *= r
            end
        end

        γ = sigmoid.( h - s*∇L )
        for t in 1:T
            # if sum(γ[t, :]) > γtot_each_t
            #     γ[t, :] *= 0.99 * γtot_each_t / sum(γ[t, :])
            # end
            γ[t, :] *= 0.99 * γtot_each_t / sum(γ[t, :])
        end
        γ = min.(max.(γ, 0), 1)
        
        sum_γ = sum(γ)
        ∇L_norm = sum(∇L .^ 2)
        println("At step = $(step), s = $(s), min_γ = $(minimum(γ)), max_γ = $(maximum(γ)), sum_γ = $(sum_γ), |∇L|^2 = $(∇L_norm), L = $(L_temp).")

        ## Saving logs:
        if save_log
            opt_log[step, :] = [L, sum_γ, ∇L_norm]
        end

        if ∇L_norm < 1e-6
            break
        end

    end

    o2 = objf_func(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)

    println("\n")
    println("minimum and maximum of γ: $(minimum(γ)), $(maximum(γ))\n")
    println("sum γ before and after optimization: $(γtot_each_t * T), $(sum(γ))\n")
    println("obj before and after optimization: $(o1), $(o2)\n")

    return o1, o2, γ, opt_log
end


"""
Pick the nodes with highest value of γ to protect.
"""
function round_up_γ(T, no_of_nodes, γ, γtot::Int)
    ind = sortperm(vec(γ), rev=true)
    nodes_to_protect = ind[1:γtot]
    γ_soln = zeros(T*no_of_nodes)
    γ_soln[ nodes_to_protect ] .= 1         ## set the γ value of the nodes to protect to be 1
    γ_soln = reshape(γ_soln, T, no_of_nodes)
    return γ_soln
end


"""
Pick the nodes with highest value of γ to protect. 
Resources are deployed in an online manner.
"""
function round_up_γ_online_resource(T, no_of_nodes, γ, γtot_each_t::Int)
    γ_soln = zero(γ)
    for t in 1:T
        ind = sortperm(γ[t, :], rev=true)
        nodes_to_protect = ind[1:γtot_each_t]
        γ_soln[ t, nodes_to_protect ] .= 1      ## set the γ value of the nodes to protect to be 1
    end
    return γ_soln
end
