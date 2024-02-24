"""
Functions for Monte Carlo (MC) simulation on the SIRP + LTM model.

σ0[i] = 0, 1  <=> node i of layer a is in state S, I
that is, σ0[i]=1 indicates that node i is one of the initially infected nodes (i.e., seeds).
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random


"""
MC simulation of SIRP + LTM.
σ_a[t, i] = 0, 1, 2, 3 <=> S, I, R, P (Protected)
σ_b[t, i] = 0, 1       <=> N (Normal), F (Failed)
"""
function SIRP_LTM_MC_sim(T::Int, adj_mat_a::SparseMatrixCSC{Int, Int}, adj_n_a::Array{Int, 2}, deg_a::Array{Int},
        σ0::Array{Float64, 1}, β::SparseMatrixCSC{Float64, Int}, γ::Array{Float64, 2}, μ::Array{Float64, 1},
        adj_mat_b::SparseMatrixCSC{Int, Int}, adj_n_b::Array{Int, 2}, deg_b::Array{Int},
        b::SparseMatrixCSC{Float64, Int}, θ::Array{Float64, 1})
    
    no_trials::Int = 100000         ## increase no. of trials for better statistical accuracy
    Random.seed!(100)

    no_of_nodes_a = size(deg_a, 1)
    no_of_nodes_b = size(deg_b, 1)
    σ_a::Array{Int, 2} = zeros(Int, T+1, no_of_nodes_a)       ## for SIRP
    σ_b::Array{Int, 2} = zeros(Int, T+1, no_of_nodes_b)       ## for LTM
    PS::Array{Float64, 2} = zeros(Float64, T+1, no_of_nodes_a)
    PI::Array{Float64, 2} = zeros(Float64, T+1, no_of_nodes_a)
    PR::Array{Float64, 2} = zeros(Float64, T+1, no_of_nodes_a)
    PP::Array{Float64, 2} = zeros(Float64, T+1, no_of_nodes_a)
    PF::Array{Float64, 2} = zeros(Float64, T+1, no_of_nodes_b)

    for step in 1:no_trials
        ## ---- SIRP process ---- ##
        ## Initial activities:
        σ_a = zeros(Int, T+1, no_of_nodes_a)
        for i in 1:no_of_nodes_a
            if rand() < σ0[i]
                σ_a[1, i] = 1
            else
                σ_a[1, i] = 0
            end
        end

        ## Activities afterwards:
        for t in 2:T+1
            ## Spontaneous protection:
            σ_a[t, :] = copy(σ_a[t-1, :])
            for i in 1:no_of_nodes_a
                if σ_a[t, i] == 0 && rand() < γ[t-1, i]
                    σ_a[t, i] = 3
                end
            end

            ## Disease transmission:
            for i in 1:no_of_nodes_a
                if σ_a[t-1, i] == 1
                    for m in 1:deg_a[i]
                        j = adj_n_a[i, m]
                        if σ_a[t, j] == 0 && rand() < β[i, j]
                            σ_a[t, j] = 1
                        end
                    end
                end
            end

            ## Recovery:
            for i in 1:no_of_nodes_a
                if σ_a[t, i] == 1 && σ_a[t-1, i] != 0 && rand() < μ[i]
                    σ_a[t, i] = 2
                end
            end
        end

        ## Stats:
        for t in 1:T+1
            for i in 1:no_of_nodes_a
                if σ_a[t, i] == 0
                    PS[t, i] += 1
                elseif σ_a[t, i] == 1
                    PI[t, i] += 1
                elseif σ_a[t, i] == 2
                    PR[t, i] += 1
                elseif σ_a[t, i] == 3
                    PP[t, i] += 1
                end
            end
        end

        ## ---- LTM process ---- ##
        ## Initial activities:
        σ_b = zeros(Int, T+1, no_of_nodes_b)
        for i in 1:no_of_nodes_b
            if σ_a[1, i] == 1 || σ_a[1, i] == 2
                σ_b[1, i] = 1
            else
                σ_b[1, i] = 0
            end
        end

        ## Activities afterwards:
        for t in 2:T+1
            ## Cascade:
            σ_b[t, :] = copy(σ_b[t-1, :])
            for i in 1:no_of_nodes_b
                if σ_b[t-1, i] == 0
                    if σ_a[t, i] == 1 || σ_a[t, i] == 2    ## failure due to infection
                        σ_b[t, i] = 1
                    else
                        temp = 0.
                        for m in 1:deg_b[i]
                            j = adj_n_b[i, m]
                            temp += b[j, i] * σ_b[t-1, j]
                        end
                        if temp >= θ[i]
                            σ_b[t, i] = 1
                        end
                    end
                end
            end
        end

        ## Stats:
        for t in 1:T+1
            for i in 1:no_of_nodes_b
                if σ_b[t, i] == 1
                    PF[t, i] += 1
                end
            end
        end
    end

    PS /= no_trials
    PI /= no_trials
    PR /= no_trials
    PP /= no_trials
    PF /= no_trials

    return PS, PI, PR, PP, PF
end
