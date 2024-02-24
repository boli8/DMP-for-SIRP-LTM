module GraphUtil
"""
Utility function for graphs.
Nodes (vertices) are strictly indexed as 1:size(nv(graph)),
edges (links) are strictly indexed as 1:size(ne(graph)).
"""


using DataFrames
using CSV
using LightGraphs
using SparseArrays

export end_node, read_graph_from_csv, write_graph_to_csv, undigraph_repr,
       construct_nonbacktrack_matrix, construct_nonbacktrack_matrix_BP_convention

function end_node(edge_list::DataFrame, e::Int, i::Int)::Int
    """For edge e->(h, t), given one end i, return the other end."""
    h, t = edge_list[e, :]
    if h == i
        j = t
    elseif t == i
        j = h
    else
        j = -1  # to indicate error
    end
    return j
end


function read_graph_from_csv(graph_name::String, directed::Bool=false)
    node_data = CSV.read(graph_name * "_nodes.csv", DataFrame)     # ::DataFrame
    no_of_nodes::Int = size(node_data, 1)

    if directed
        edge_data = CSV.read(graph_name * "_diedges.csv", DataFrame)
        graph = SimpleDiGraph(no_of_nodes)
    else
        edge_data = CSV.read(graph_name * "_edges.csv", DataFrame)
        graph = SimpleGraph(no_of_nodes)
    end

    for e in 1:size(edge_data, 1)
        i, j = edge_data[e, 2], edge_data[e, 3]
        add_edge!(graph, i, j)
    end

    return node_data, edge_data, graph
end


function write_graph_to_csv(graph::AbstractGraph, graph_name::String, directed::Bool=false)
    """Given a SimpleGraph or a SimpleDiGraph, write a graph to csv file.
    When given a SimpleGraph but wants to write a di-graph, assumes both directions are present for an un-directed edge."""
    open(graph_name * "_nodes.csv", "w") do io
        write(io, "node,node_name\n")
        for (xi, i) in enumerate(vertices(graph))
            write(io, "$xi,$i\n")
        end
    end

    if directed
        edge_filename = graph_name * "_diedges.csv"
    else
        edge_filename = graph_name * "_edges.csv"
    end

    open(edge_filename, "w") do io
        write(io, "edge,from_node,to_node\n")
        diedge_count = 1
        for (e, edg) in enumerate(edges(graph))
            i, j = src(edg), dst(edg)
            if typeof(graph) <: SimpleGraph && directed # input an un-digraph, but want to write a digraph
                write(io, "$diedge_count,$i,$j\n")
                write(io, "$(diedge_count+1),$j,$i\n")
                diedge_count += 2
            else
                write(io, "$e,$i,$j\n")
            end
        end
    end
    return nothing
end


function undigraph_repr(graph::SimpleGraph, edge_data::DataFrame)
    """Representation of un-directed graphs."""
    max_deg::Int = 0     # def as maximum degree
    deg::Array{Int} = zeros(nv(graph))
    for i in vertices(graph)
        deg[i] = degree(graph, i)   # different from digraph
        if deg[i] > max_deg
            max_deg = deg[i]
        end
    end

    edge_list::DataFrame = edge_data[:, 2:3]
    edge_indx::Dict = Dict(vcat(
                            [(edge_list[e, 1], edge_list[e, 2]) => e for e in 1:size(edge_list, 1)],
                            [(edge_list[e, 2], edge_list[e, 1]) => e for e in 1:size(edge_list, 1)]
                            ))   # index of an edge in edge_list, different from digraph
    adj_n::Array{Int} = zeros(Int, (nv(graph), max_deg))    # node adjacency list
    adj_e::Array{Int} = zeros(Int, (nv(graph), max_deg))    # edge adjacency list
    adj_e_indx::Dict{Tuple{Int, Int}, Int} = Dict()     # index of an edge in adj_e
    B::SparseMatrixCSC{Int, Int} = spzeros(Int, nv(graph), ne(graph))   # usual incidence matrix for un-directed graph
    adj_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, nv(graph), nv(graph))   # usual adjacency matrix for un-directed graph

    for i in vertices(graph)
        n = 1   # the index of edge e in the list of edges adjacent to node i
        for j in neighbors(graph, i)
            e = edge_indx[(i, j)]
            adj_n[i, n] = j
            adj_e[i, n] = e
            adj_e_indx[(i, e)] = n
            adj_mat[i, j] = 1
            n += 1
            if i == edge_list[e, 1]
                B[i, e] = -1     # i = head(e)
            else
                B[i, e] = 1    # i = tail(e)
            end
        end
    end

    return max_deg, deg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, adj_mat, B
end


function construct_nonbacktrack_matrix(graph::SimpleGraph, edge_data::DataFrame)
    """Construct the nonbactracking matrix of undirected graph. Unweighted version.
    Index: 1 : no_of_edges -> diedges of an assumed orientation,
    no_of_nodes+1 : end -> diedges of the reversed orientation."""
    no_of_nodes::Int = nv(graph)
    no_of_edges::Int = ne(graph)
    edge_list::DataFrame = edge_data[:, 2:3]

    diedge_list::DataFrame = edge_data[:, 2:3]
    for e in 1:no_of_edges
        push!( diedge_list, [ edge_list[e, 2], edge_list[e, 1] ] )
    end
    ## index of an edge in diedge_list:
    diedge_indx::Dict = Dict( [(diedge_list[e, 1], diedge_list[e, 2]) => e for e in 1:2*no_of_edges] )

    NB1::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)
    NB2::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)
    S::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, 2*no_of_edges)
    T::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, 2*no_of_edges)
    J::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)

    for e in 1:2*no_of_edges
        ep = mod(e + no_of_edges - 1, 2*no_of_edges) + 1
        J[e, ep] = 1

        i, j = diedge_list[e, 1], diedge_list[e, 2]
        for k in 1:no_of_nodes
            if k == i
                S[k, e] = 1
            elseif k == j
                T[k, e] = 1
            end
        end
    end
    NB1 = T' * S - J


    for e in 1:2*no_of_edges
        i, j = diedge_list[e, 1], diedge_list[e, 2]
        for ep in 1:2*no_of_edges
            k, l = diedge_list[ep, 1], diedge_list[ep, 2]
            if j==k && i!=l
                NB2[e, ep] = 1
            end
        end
    end


    adj_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    deg_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    id_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    zero_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    IB_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_nodes, 2*no_of_nodes)  ## Ihara-Bass formula
    for i in vertices(graph)
        id_mat[i, i] = 1
        deg_mat[i, i] = degree(graph, i)
        for j in neighbors(graph, i)
            adj_mat[i, j] = 1
        end
    end

    IB_mat = [adj_mat  id_mat-deg_mat; id_mat  zero_mat]

    return diedge_list, S, T, J, NB1, NB2, IB_mat
end


function construct_nonbacktrack_matrix_BP_convention(graph::SimpleGraph, edge_data::DataFrame)
    """Construct the nonbactracking matrix of undirected graph. Unweighted version.
    Index: 1 : no_of_edges -> diedges of an assumed orientation,
    no_of_nodes+1 : end -> diedges of the reversed orientation.
    Convention of stability of belief propagation, B[e1, e2], the walk is from e2 to e1."""
    no_of_nodes::Int = nv(graph)
    no_of_edges::Int = ne(graph)
    edge_list::DataFrame = edge_data[:, 2:3]

    diedge_list::DataFrame = edge_data[:, 2:3]
    for e in 1:no_of_edges
        push!( diedge_list, [ edge_list[e, 2], edge_list[e, 1] ] )
    end
    ## index of an edge in diedge_list:
    diedge_indx::Dict = Dict( [(diedge_list[e, 1], diedge_list[e, 2]) => e for e in 1:2*no_of_edges] )

    NB1::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)
    NB2::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)
    S::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, 2*no_of_edges)
    T::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, 2*no_of_edges)
    J::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_edges, 2*no_of_edges)

    for e in 1:2*no_of_edges
        ep = mod(e + no_of_edges - 1, 2*no_of_edges) + 1
        J[e, ep] = 1

        i, j = diedge_list[e, 1], diedge_list[e, 2]
        for k in 1:no_of_nodes
            if k == i
                S[k, e] = 1
            elseif k == j
                T[k, e] = 1
            end
        end
    end
    NB1 = S' * T - J


    for e in 1:2*no_of_edges
        i, j = diedge_list[e, 1], diedge_list[e, 2]
        for ep in 1:2*no_of_edges
            k, l = diedge_list[ep, 1], diedge_list[ep, 2]
            if i==l && j!=k
                NB2[e, ep] = 1
            end
        end
    end


    adj_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    deg_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    id_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    zero_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, no_of_nodes, no_of_nodes)
    IB_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, 2*no_of_nodes, 2*no_of_nodes)  ## Ihara-Bass formula
    for i in vertices(graph)
        id_mat[i, i] = 1
        deg_mat[i, i] = degree(graph, i)
        for j in neighbors(graph, i)
            adj_mat[i, j] = 1
        end
    end

    IB_mat = [zero_mat  deg_mat-id_mat; -id_mat  adj_mat]

    return diedge_list, S, T, J, NB1, NB2, IB_mat
end


end  # module GraphUtil
