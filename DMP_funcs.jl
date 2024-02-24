"""
Functions for Dynamic message-passing on the SIRP + LTM model.

σ0[i] = 0, 1  <=> node i of layer a is in state S, I
that is, σ0[i]=1 indicates that node i is one of the initially infected nodes (i.e., seeds).
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random


sigmoid(x::Real) = one(x) / (one(x) + exp(-x))


inv_sigmoid(y::Real) = log( y / ( one(y) - y ) )


"""
Softmax function.
The input h should be a d-dimensional array.
"""
function softmax(h)
    hmax = maximum(h)
    Ph = exp.( h .- hmax )
    Ph = Ph ./ sum(Ph)
    return Ph
end


"""
Convert an index to a Boolean configuration.
γ: index counting from 1 (assume Int64).
nvar: number of variables (must not exceed 64).
"""
function index_to_config(γ::Int, nvar::Int)
    if nvar > 64
        throw(DomainError(nvar, "nvar cannot be greater than 64."))
    end

    s = zeros(Int, nvar)
    x = collect(bitstring(γ - 1))
    for i in 1:nvar
        s[i] = parse(Int, x[end-nvar+i])
    end
    return s
end


"""
Representation of two-layer network.
"""
function build_two_layer_net(adj_n_a, deg_a, adj_n_b, deg_b)
    if length(deg_a) != length(deg_b)
        println("The two network must have the same sizes.")
        return 0
    end

    no_of_nodes = length(deg_a)
    max_deg_c = maximum(deg_a) + maximum(deg_b)

    adj_n_Uab = zeros(Int, no_of_nodes, max_deg_c)        ## neighbors in either a or b
    adj_n_Λab = zeros(Int, no_of_nodes, max_deg_c)        ## neighbors in both a and b
    adj_n_Ea = zeros(Int, no_of_nodes, maximum(deg_a))    ## neighbors exclusive in layer a
    adj_n_Eb = zeros(Int, no_of_nodes, maximum(deg_b))    ## neighbors exclusive in layer b

    deg_Uab = zeros(Int, no_of_nodes)
    deg_Λab = zeros(Int, no_of_nodes)
    deg_Ea = zeros(Int, no_of_nodes)
    deg_Eb = zeros(Int, no_of_nodes)

    for i in 1:no_of_nodes
        ∂i_a = adj_n_a[i, 1:deg_a[i]]
        ∂i_b = adj_n_b[i, 1:deg_b[i]]

        Uab = union(∂i_a, ∂i_b)
        deg_Uab[i] = length(Uab)
        adj_n_Uab[i, 1:deg_Uab[i]] = Uab[:]

        Λab = intersect(∂i_a, ∂i_b)
        deg_Λab[i] = length(Λab)
        adj_n_Λab[i, 1:deg_Λab[i]] = Λab[:]

        Ea = setdiff(∂i_a, ∂i_b)
        deg_Ea[i] = length(Ea)
        adj_n_Ea[i, 1:deg_Ea[i]] = Ea[:]

        Eb = setdiff(∂i_b, ∂i_a)
        deg_Eb[i] = length(Eb)
        adj_n_Eb[i, 1:deg_Eb[i]] = Eb[:]
    end

    return adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb
end


"""
Dynamic message passing of SIRP.
----
βv is a vector containing the infection probabilities with βv[e] = β[i, j],
where the e-th edge is (i, j).
----
opt is a string indicating which variables are the control variables; this is important
as the type of the control variables need to propagate to the dynamical variables
for the effectiveness of autodiff.
"""
function cal_dynamic_messages_SIRP(T, edge_list, adj_n, deg, σ0, βv, γ, μ, opt)
    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "gamma"
        type_of_var = eltype(γ)
    end

    no_of_nodes = size(deg, 1)
    no_of_edges = size(edge_list, 1)

    β = spzeros(eltype(βv), no_of_nodes, no_of_nodes)   ## N X N sparse matrix

    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        β[i, j] = βv[e]
        β[j, i] = βv[e]
    end

    PS_cav = [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]
    PI_cav = [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]
    PR_cav = [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]
    PP_cav = [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]
    θ_cav =  [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]
    ϕ_cav =  [spzeros(type_of_var, no_of_nodes, no_of_nodes) for t in 1:T+1]

    ## Initial condition:
    for i in 1:no_of_nodes
        for n in 1:deg[i]
            j = adj_n[i, n]
            θ_cav[1][i, j] = 1.
            ϕ_cav[1][i, j] = σ0[i]
            PS_cav[1][i, j] = (1 - σ0[i])
            PI_cav[1][i, j] = σ0[i]
            PR_cav[1][i, j] = 0
        end
    end

    dγ_temp = ones(type_of_var, T+1, no_of_nodes)
    for t in 2:T+1
        for i in 1:no_of_nodes
            dγ_temp[t, i] = dγ_temp[t-1, i] * (1 - γ[t-1, i])

            for n in 1:deg[i]
                j = adj_n[i, n]
                temp = 1.
                for m in 1:deg[i]
                    k = adj_n[i, m]
                    if k == j
                        continue
                    end
                    θ_cav[t][k, i] = θ_cav[t-1][k, i] - β[k, i] * ϕ_cav[t-1][k, i]
                    temp *= θ_cav[t][k, i]
                end
                PS_cav[t][i, j] = (1 - σ0[i]) * dγ_temp[t, i] * temp

                PR_cav[t][i, j] = PR_cav[t-1][i, j] + μ[i] * PI_cav[t-1][i, j]
                PP_cav[t][i, j] = PP_cav[t-1][i, j] + γ[t-1, i] * PS_cav[t-1][i, j]
                PI_cav[t][i, j] = 1 - PR_cav[t][i, j] - PS_cav[t][i, j] - PP_cav[t][i, j]
                ϕ_cav[t][i, j] = (1 - β[i, j] - μ[i] + β[i, j]*μ[i]) * ϕ_cav[t-1][i, j] -
                                 ( PS_cav[t][i, j] - PS_cav[t-1][i, j]*(1 - γ[t-1, i]) )
                θ_cav[t][i, j] = θ_cav[t-1][i, j] - β[i, j] * ϕ_cav[t-1][i, j]
            end
        end
    end

    return PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav
end


"""
Marginal probability of DMP in the SIRP layer.
----
'opt' is a string indicating which variables are the control variables; this is important
as the type of the control variables need to propagate to the dynamical variables
for the effectiveness of autodiff.
"""
function cal_DMP_marginal_SIRP(T, adj_n, deg, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)
    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "gamma"
        type_of_var = eltype(γ)
    end

    no_of_nodes = size(deg, 1)

    PS_mgn = zeros(type_of_var, T+1, no_of_nodes)
    PI_mgn = zeros(type_of_var, T+1, no_of_nodes)
    PR_mgn = zeros(type_of_var, T+1, no_of_nodes)
    PP_mgn = zeros(type_of_var, T+1, no_of_nodes)

    ## Initial condition:
    for i in 1:no_of_nodes
        PS_mgn[1, i] = 1 - σ0[i]
        PI_mgn[1, i] = σ0[i]
        PR_mgn[1, i] = 0
        PP_mgn[1, i] = 0
    end

    dγ_temp = ones(type_of_var, T+1, no_of_nodes)
    for t in 2:T+1
        for i in 1:no_of_nodes
            dγ_temp[t, i] = dγ_temp[t-1, i] * (1 - γ[t-1, i])

            temp = 1.
            for m in 1:deg[i]
                k = adj_n[i, m]
                temp *= θ_cav[t][k, i]
            end
            PS_mgn[t, i] = (1 - σ0[i]) * dγ_temp[t, i] * temp
            PR_mgn[t, i] = PR_mgn[t-1, i] + μ[i] * PI_mgn[t-1, i]
            PP_mgn[t, i] = PP_mgn[t-1, i] + γ[t-1, i] * PS_mgn[t-1, i]
            PI_mgn[t, i] = 1 - PS_mgn[t, i] - PR_mgn[t, i] - PP_mgn[t, i]
        end
    end

    return PS_mgn, PI_mgn, PR_mgn, PP_mgn
end


"""
Dynamic message passing of the LTM layer.
"""
function cal_dynamic_messages_LTM_by_PS_mgn(T, adj_mat, adj_n, deg, σ0, b, θ, PS_mgn, opt)
    if opt == "beta" || opt == "mu" || opt == "gamma"
        type_of_var = eltype(PS_mgn)
    end

    no_of_nodes = size(adj_mat, 1)
    PF_cav = [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]

    ## Initial condition:
    for i in 1:no_of_nodes
        for n in 1:deg[i]
            j = adj_n[i, n]
            PF_cav[1][i, j] = σ0[i]
        end
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            for n in 1:deg[i]
                j = adj_n[i, n]

                temp = 0.
                for indx in 1:2^(deg[i]-1)
                    x = index_to_config(indx, deg[i]-1)

                    nk = 1
                    sum_bx = 0.
                    for m in 1:deg[i]
                        k = adj_n[i, m]
                        if k == j
                            continue
                        end
                        sum_bx += b[k, i]*x[nk]
                        nk += 1
                    end

                    if sum_bx >= θ[i]
                        nk = 1
                        prod_PF = 1.
                        for m in 1:deg[i]
                            k = adj_n[i, m]
                            if k == j
                                continue
                            end

                            if x[nk] == 1
                                prod_PF *= PF_cav[t-1][k, i]
                            elseif x[nk] == 0
                                prod_PF *= (1 - PF_cav[t-1][k, i])
                            end
                            nk += 1
                        end
                        temp += prod_PF
                    end

                end # of indx
                PF_cav[t][i, j] = (1 - PS_mgn[t, i]) + PS_mgn[t, i] * temp

            end # of n
        end # of i
    end # of t

    return PF_cav
end


"""
Marginal probability of DMP in the LTM layer.
"""
function cal_DMP_marginal_LTM_by_PS_mgn(T, adj_mat, adj_n, deg, σ0, b, θ, PS_mgn, PF_cav, opt)
    if opt == "beta" || opt == "mu" || opt == "gamma"
        type_of_var = eltype(PS_mgn)
    end

    no_of_nodes = size(adj_mat, 1)
    PF_mgn = zeros(type_of_var, T+1, no_of_nodes)

    ## Initial condition:
    for i in 1:no_of_nodes
        PF_mgn[1, i] = 1 - PS_mgn[1, i]
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            temp = 0.
            for indx in 1:2^(deg[i])
                x = index_to_config(indx, deg[i])

                sum_bx = 0.
                for m in 1:deg[i]
                    k = adj_n[i, m]
                    sum_bx += b[k, i] * x[m]
                end

                if sum_bx >= θ[i]
                    prod_PF = 1.
                    for m in 1:deg[i]
                        k = adj_n[i, m]
                        if x[m] == 1
                            prod_PF *= PF_cav[t-1][k, i]
                        elseif x[m] == 0
                            prod_PF *= (1 - PF_cav[t-1][k, i])
                        end
                    end
                    temp += prod_PF
                end
            end # of indx
            PF_mgn[t, i] = (1 - PS_mgn[t, i]) + PS_mgn[t, i] * temp

        end # of i
    end # of t

    return PF_mgn
end


"""
Dynamic message passing of LTM (layer b) induced by SIRP (layer a).
Full expression for the LTM layer.
"""
function cal_dynamic_messages_LTM_SIRP_full(T, edge_list_a, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
            PS_cav, PI_cav, PR_cav, θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, opt)
    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "gamma"
        type_of_var = eltype(γ)
    end

    no_of_nodes_a = size(deg_a, 1)
    no_of_edges_a = size(edge_list_a, 1)
    ψ_cav = [spzeros(type_of_var, no_of_nodes_a, no_of_nodes_a) for t in 1:T+1]
    ψm_cav = [spzeros(type_of_var, no_of_nodes_a, no_of_nodes_a) for t in 1:T+1, tp in 1:T+1]
    dγ_temp = ones(Float64, T+1, no_of_nodes_a)

    no_of_nodes_b = size(deg_b, 1)
    χ_cav = [spzeros(type_of_var, no_of_nodes_b, no_of_nodes_b) for t in 1:T+1]
    χm_cav = [spzeros(type_of_var, no_of_nodes_b, no_of_nodes_b) for t in 1:T+1, tp in 1:T+1]
    PF_cav = [spzeros(type_of_var, no_of_nodes_b, no_of_nodes_b) for t in 1:T+1]
    PSF_cav = [spzeros(type_of_var, no_of_nodes_b, no_of_nodes_b) for t in 1:T+1]
    PPF_cav = [spzeros(type_of_var, no_of_nodes_b, no_of_nodes_b) for t in 1:T+1]

    ## β matrix:
    β = spzeros(eltype(βv), no_of_nodes_a, no_of_nodes_a)   ## N X N sparse matrix
    for e in 1:no_of_edges_a
        i, j = edge_list_a[e, :]
        β[i, j] = βv[e]
        β[j, i] = βv[e]
    end

    ## Cumulative (1-γ):
    for t in 2:T+1
        for i in 1:no_of_nodes_a
            dγ_temp[t, i] = dγ_temp[t-1, i] * (1 - γ[t-1, i])
        end
    end

    ## Evolution of ψ_cav:
    for i in 1:length(deg_Λab)
        for n in 1:deg_Λab[i]
            j = adj_n_Λab[i, n]
            ψ_cav[2][i, j] = (1 - β[i, j]) * PI_mgn[1, i]
        end
    end
    ##
    for t in 3:T+1
        for i in 1:length(deg_Λab)
            for n in 1:deg_Λab[i]
                j = adj_n_Λab[i, n]
                ψ_cav[t][i, j] = ψ_cav[t-1][i, j] - β[i, j]*ϕ_cav[t-1][i, j] +
                                 (1 - γ[t-2, i]) * PS_cav[t-2][i, j] - PS_cav[t-1][i, j]
            end
        end
    end

    ## Evolution of ψm_cav:
    for i in 1:length(deg_Λab)
        for n in 1:deg_Λab[i]
            j = adj_n_Λab[i, n]
            ψm_cav[2, 2][i, j] = PI_mgn[1, i]
        end
    end
    ##
    for t in 3:T+1
        # case of initial ε:
        ε = 2
        for i in 1:length(deg_Λab)
            for n in 1:deg_Λab[i]
                j = adj_n_Λab[i, n]
                ψm_cav[t, ε][i, j] = PI_cav[t-1][i, j] + PR_cav[t-1][i, j]
            end
        end
        ##
        for ε in 3:t
            for i in 1:length(deg_Λab)
                for n in 1:deg_Λab[i]
                    j = adj_n_Λab[i, n]
                    ψm_cav[t, ε][i, j] = ψ_cav[ε-1][i, j] + PI_cav[t-1][i, j] + PR_cav[t-1][i, j] -
                                         (PI_cav[ε-2][i, j] + PR_cav[ε-2][i, j])
                end
            end
        end
    end

    ## Evolution of χ_cav and PF_cav and others:
    for i in 1:length(deg_Eb)
        for n in 1:deg_Eb[i]
            j = adj_n_Eb[i, n]
            PF_cav[1][i, j] = PI_mgn[1, i] + PR_mgn[1, i]
            PSF_cav[1][i, j] = 0
            PPF_cav[1][i, j] = 0
        end
    end
    for i in 1:length(deg_Λab)
        for n in 1:deg_Λab[i]
            j = adj_n_Λab[i, n]
            χ_cav[2][i, j] = ψ_cav[2][i, j]
            PSF_cav[1][i, j] = 0
            PPF_cav[1][i, j] = 0
            χm_cav[2, 2][i, j] = ψm_cav[2, 2][i, j]
        end
    end
    ##
    for t in 2:T+1
        for i in 1:no_of_nodes_b
            for n in 1:deg_b[i]
                j = adj_n_b[i, n]

                ##---- begin PSF_cav ----##
                temp_θ_cav = PS_mgn[1, i]
                for m in 1:deg_Ea[i]
                    k = adj_n_Ea[i, m]
                    if k == j
                        continue
                    end
                    temp_θ_cav *= θ_cav[t][k, i]
                end

                temp = 0.
                for indx in 1:2^(deg_b[i]-1)
                    x = index_to_config(indx, deg_b[i]-1)

                    nk = 1
                    sum_bx = 0.
                    ## Case k in ∂i_b \ ∂i_a Λ ∂i_b Λ j:
                    for m in 1:deg_Eb[i]
                        k = adj_n_Eb[i, m]
                        if k == j
                            continue
                        end
                        sum_bx += b[k, i]*x[nk]
                        nk += 1
                    end
                    ## Case k in ∂i_a Λ ∂i_b \ j:
                    for m in 1:deg_Λab[i]
                        k = adj_n_Λab[i, m]
                        if k == j
                            continue
                        end
                        sum_bx += b[k, i]*x[nk]
                        nk += 1
                    end

                    if sum_bx >= θ[i]
                        nk = 1
                        prod_PF = 1.

                        ## Case k in ∂i_b \ ∂i_a Λ ∂i_b Λ j:
                        for m in 1:deg_Eb[i]
                            k = adj_n_Eb[i, m]
                            if k == j
                                continue
                            end

                            if x[nk] == 1
                                prod_PF *= PF_cav[t-1][k, i]
                            elseif x[nk] == 0
                                prod_PF *= (1 - PF_cav[t-1][k, i])
                            end
                            nk += 1
                        end

                        ## Case k in ∂i_a Λ ∂i_b \ j:
                        for m in 1:deg_Λab[i]
                            k = adj_n_Λab[i, m]
                            if k == j
                                continue
                            end

                            if x[nk] == 1
                                prod_PF *= χ_cav[t][k, i]
                            elseif x[nk] == 0
                                prod_PF *= (θ_cav[t][k, i] - χ_cav[t][k, i])
                            end
                            nk += 1
                        end

                        temp += prod_PF
                    end

                end # of indx

                PSF_cav[t][i, j] = dγ_temp[t, i] * temp_θ_cav * temp
                ##---- end PSF_cav ----##


                ##---- begin PPF_cav ----##
                for ε in 2:t
                    temp_θ_cav = PS_mgn[1, i]
                    for m in 1:deg_Ea[i]
                        k = adj_n_Ea[i, m]
                        if k == j
                            continue
                        end
                        temp_θ_cav *= θ_cav[ε-1][k, i]
                    end

                    temp = 0.
                    for indx in 1:2^(deg_b[i]-1)
                        x = index_to_config(indx, deg_b[i]-1)

                        nk = 1
                        sum_bx = 0.
                        ## Case k in ∂i_b \ ∂i_a Λ ∂i_b Λ j:
                        for m in 1:deg_Eb[i]
                            k = adj_n_Eb[i, m]
                            if k == j
                                continue
                            end
                            sum_bx += b[k, i]*x[nk]
                            nk += 1
                        end
                        ## Case k in ∂i_a Λ ∂i_b \ j:
                        for m in 1:deg_Λab[i]
                            k = adj_n_Λab[i, m]
                            if k == j
                                continue
                            end
                            sum_bx += b[k, i]*x[nk]
                            nk += 1
                        end

                        if sum_bx >= θ[i]
                            nk = 1
                            prod_PF = 1.

                            ## Case k in ∂i_b \ ∂i_a Λ ∂i_b Λ j:
                            for m in 1:deg_Eb[i]
                                k = adj_n_Eb[i, m]
                                if k == j
                                    continue
                                end

                                if x[nk] == 1
                                    prod_PF *= PF_cav[t-1][k, i]
                                elseif x[nk] == 0
                                    prod_PF *= (1 - PF_cav[t-1][k, i])
                                end
                                nk += 1
                            end

                            ## Case k in ∂i_a Λ ∂i_b \ j:
                            for m in 1:deg_Λab[i]
                                k = adj_n_Λab[i, m]
                                if k == j
                                    continue
                                end

                                if x[nk] == 1
                                    prod_PF *= χm_cav[t, ε][k, i]
                                elseif x[nk] == 0
                                    prod_PF *= (θ_cav[ε-1][k, i] - χm_cav[t, ε][k, i])
                                end
                                nk += 1
                            end

                            temp += prod_PF
                        end

                    end # of indx

                    PPF_cav[t][i, j] += γ[ε-1, i] * dγ_temp[ε-1, i] * temp_θ_cav * temp
                end
                ##---- end PPF_cav ----##

                ## Update cavity probabilities:
                if j in adj_n_Λab[i, 1:deg_Λab[i]]
                    if t < T+1
                        χ_cav[t+1][i, j] = ψ_cav[t+1][i, j] + PSF_cav[t][i, j] + PPF_cav[t][i, j]

                        for ε in 2:t+1
                            χm_cav[t+1, ε][i, j] = ψm_cav[t+1, ε][i, j] + PSF_cav[t][i, j] + PPF_cav[t][i, j]
                        end
                    end
                else
                    PF_cav[t][i, j] = PI_mgn[t, i] + PR_mgn[t, i] + PSF_cav[t][i, j] + PPF_cav[t][i, j]
                end

            end # of n
        end # of i
    end # of t

    return ψ_cav, PF_cav, χ_cav, χm_cav
end


"""
Marginal probability of LTM (layer b) induced by SIRP (layer a).
Full expression for the LTM layer.
"""
function cal_DMP_marginal_LTM_SIRP_full(T, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
            θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, ψ_cav, PF_cav, χ_cav, χm_cav, opt)
    
    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "gamma"
        type_of_var = eltype(γ)
    end

    no_of_nodes_a = size(deg_a, 1)
    dγ_temp = ones(Float64, T+1, no_of_nodes_a)

    no_of_nodes_b = size(deg_b, 1)
    PF_mgn = zeros(type_of_var, T+1, no_of_nodes_b)
    PSF_mgn = zeros(type_of_var, T+1, no_of_nodes_b)
    PPF_mgn = zeros(type_of_var, T+1, no_of_nodes_b)

    ## Cumulative (1-γ):
    for t in 2:T+1
        for i in 1:no_of_nodes_a
            dγ_temp[t, i] = dγ_temp[t-1, i] * (1 - γ[t-1, i])
        end
    end

    ## Evolution of PF_mgn
    for i in 1:no_of_nodes_a
        PF_mgn[1, i] = PI_mgn[1, i] + PR_mgn[1, i]
        PSF_mgn[1, i] = 0
        PPF_mgn[1, i] = 0
    end
    ##
    for t in 2:T+1
        for i in 1:no_of_nodes_b

            ##---- begin PSF_mgn ----##
            temp_θ_cav = PS_mgn[1, i]
            for m in 1:deg_Ea[i]
                k = adj_n_Ea[i, m]
                temp_θ_cav *= θ_cav[t][k, i]
            end

            temp = 0.
            for indx in 1:2^(deg_b[i])
                x = index_to_config(indx, deg_b[i])

                nk = 1
                sum_bx = 0.
                ## Case k in ∂i_b \ ∂i_a Λ ∂i_b:
                for m in 1:deg_Eb[i]
                    k = adj_n_Eb[i, m]
                    sum_bx += b[k, i]*x[nk]
                    nk += 1
                end
                ## Case k in ∂i_a Λ ∂i_b:
                for m in 1:deg_Λab[i]
                    k = adj_n_Λab[i, m]
                    sum_bx += b[k, i]*x[nk]
                    nk += 1
                end

                if sum_bx >= θ[i]
                    nk = 1
                    prod_PF = 1.

                    ## Case k in ∂i_b \ ∂i_a Λ ∂i_b:
                    for m in 1:deg_Eb[i]
                        k = adj_n_Eb[i, m]
                        if x[nk] == 1
                            prod_PF *= PF_cav[t-1][k, i]
                        elseif x[nk] == 0
                            prod_PF *= (1 - PF_cav[t-1][k, i])
                        end
                        nk += 1
                    end

                    ## Case k in ∂i_a Λ ∂i_b:
                    for m in 1:deg_Λab[i]
                        k = adj_n_Λab[i, m]
                        if x[nk] == 1
                            prod_PF *= χ_cav[t][k, i]
                        elseif x[nk] == 0
                            prod_PF *= (θ_cav[t][k, i] - χ_cav[t][k, i])
                        end
                        nk += 1
                    end

                    temp += prod_PF
                end

            end # of indx

            PSF_mgn[t, i] = dγ_temp[t, i] * temp_θ_cav * temp
            ##---- end PSF_mgn ----##


            ##---- begin PPF_mgn ----##
            for ε in 2:t
                temp_θ_cav = PS_mgn[1, i]
                for m in 1:deg_Ea[i]
                    k = adj_n_Ea[i, m]
                    temp_θ_cav *= θ_cav[ε-1][k, i]
                end

                temp = 0.
                for indx in 1:2^(deg_b[i])
                    x = index_to_config(indx, deg_b[i])

                    nk = 1
                    sum_bx = 0.
                    ## Case k in ∂i_b \ ∂i_a Λ ∂i_b:
                    for m in 1:deg_Eb[i]
                        k = adj_n_Eb[i, m]
                        sum_bx += b[k, i]*x[nk]
                        nk += 1
                    end
                    ## Case k in ∂i_a Λ ∂i_b:
                    for m in 1:deg_Λab[i]
                        k = adj_n_Λab[i, m]
                        sum_bx += b[k, i]*x[nk]
                        nk += 1
                    end

                    if sum_bx >= θ[i]
                        nk = 1
                        prod_PF = 1.

                        ## Case k in ∂i_b \ ∂i_a Λ ∂i_b:
                        for m in 1:deg_Eb[i]
                            k = adj_n_Eb[i, m]
                            if x[nk] == 1
                                prod_PF *= PF_cav[t-1][k, i]
                            elseif x[nk] == 0
                                prod_PF *= (1 - PF_cav[t-1][k, i])
                            end
                            nk += 1
                        end

                        ## Case k in ∂i_a Λ ∂i_b:
                        for m in 1:deg_Λab[i]
                            k = adj_n_Λab[i, m]
                            if x[nk] == 1
                                prod_PF *= χm_cav[t, ε][k, i]
                            elseif x[nk] == 0
                                prod_PF *= (θ_cav[ε-1][k, i] - χm_cav[t, ε][k, i])
                            end
                            nk += 1
                        end

                        temp += prod_PF
                    end

                end # of indx

                PPF_mgn[t, i] += γ[ε-1, i] * dγ_temp[ε-1, i] * temp_θ_cav * temp
            end
            ##---- end PPF_mgn ----##

            PF_mgn[t, i] = PI_mgn[t, i] + PR_mgn[t, i] + PSF_mgn[t, i] + PPF_mgn[t, i]

        end # of i
    end # of t

    return PF_mgn
end


"""
Probability of all states governed by the forward equations of SIRP+LTM.
Full expression for the LTM layer.
"""
function forward_all_states_SIRP_LTM(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)
    PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav = cal_dynamic_messages_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, opt)
    PS_mgn, PI_mgn, PR_mgn, PP_mgn = cal_DMP_marginal_SIRP(T, adj_n_a, deg_a, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)

    ψ_cav, PF_cav, χ_cav, χm_cav = cal_dynamic_messages_LTM_SIRP_full(T, edge_list_a, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
                PS_cav, PI_cav, PR_cav, θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, opt)
    PF_mgn = cal_DMP_marginal_LTM_SIRP_full(T, adj_n_a, deg_a, βv, γ, adj_n_b, deg_b, b, θ,
                adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb,
                θ_cav, ϕ_cav, PS_mgn, PI_mgn, PR_mgn, ψ_cav, PF_cav, χ_cav, χm_cav, opt)
    
    return PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn
end


"""
Probability of all states governed by the forward equations of SIRP+LTM.
Approximated expression for the LTM layer, assuming decorrelation of networks a and b.
"""
function forward_all_states_SIRP_LTM_approx(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, adj_n_b, deg_b, b, θ,
            adj_n_Uab, adj_n_Λab, adj_n_Ea, adj_n_Eb, deg_Uab, deg_Λab, deg_Ea, deg_Eb, opt)
    PS_cav, PI_cav, PR_cav, PP_cav, θ_cav, ϕ_cav = cal_dynamic_messages_SIRP(T, edge_list_a, adj_n_a, deg_a, σ0, βv, γ, μ, opt)
    PS_mgn, PI_mgn, PR_mgn, PP_mgn = cal_DMP_marginal_SIRP(T, adj_n_a, deg_a, σ0, βv, γ, μ, PS_cav, θ_cav, ϕ_cav, opt)

    σ0_b = PI_mgn[1, :] + PR_mgn[1, :]
    PF_cav = cal_dynamic_messages_LTM_by_PS_mgn(T, adj_mat_b, adj_n_b, deg_b, σ0_b, b, θ, PS_mgn+PP_mgn, opt)
    PF_mgn = cal_DMP_marginal_LTM_by_PS_mgn(T, adj_mat_b, adj_n_b, deg_b, σ0_b, b, θ, PS_mgn+PP_mgn, PF_cav, opt)
    
    return PS_mgn, PI_mgn, PR_mgn, PP_mgn, PF_mgn
end
