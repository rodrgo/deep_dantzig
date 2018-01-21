
# From Google OR tools
# https://developers.google.com/optimization/flow/maxflow 

from __future__ import print_function
from ortools.graph import pywrapgraph
import networkx as nx
import sys
import numpy as np

from scipy.sparse import csc_matrix
from scipy.optimize import linprog

def generate_networkx_graph(name, n=100, m=2000, p=0.2, seed=None, directed=True):
    '''
         Generate a random graph using networkx package
    '''
    # Factory method
    if name == 'toy_1':
        start_nodes = [0, 0, 0, 1, 1, 2, 2, 3, 3]
        end_nodes = [1, 2, 3, 2, 4, 3, 4, 2, 4]
        capacities = [20, 30, 10, 40, 30, 10, 20, 5, 20]
        G = nx.DiGraph()
        G.add_weighted_edges_from([(x,y,z) for x,y,z in zip(start_nodes, end_nodes, capacities)])
    elif name == 'toy_2':
        start_nodes = [0, 0, 1, 2, 1, 2, 3, 4]
        end_nodes   = [1, 2, 3, 3, 4, 4, 5, 5]
        capacities  = [5, 5, 3, 1, 6, 3, 6, 6]
        G = nx.DiGraph()
        G.add_weighted_edges_from([(x,y,z) for x,y,z in zip(start_nodes, end_nodes, capacities)])
    if name == 'gnm':
        # G(n,m) = set of all graphs with n nodes and m edges
        G = nx.gnm_random_graph(n, m, seed, directed)
    elif name == 'gnp':
        # G(n, p) = chooses each of the possible edges with probability p
        G = nx.gnp_random_graph(n, p, seed, directed)
    elif name == 'erdos_renyi':
        # G(n, p) = chooses each of the possible edges with probability p
        G = nx.erdos_renyi_graph(n, p, seed, directed)
    elif name == 'binomial':
        # G(n, p) = chooses each of the possible edges with probability p
        G = nx.binomial_graph(n, p, seed, directed)
    elif name == 'barabasi':
        # A graph of n nodes is grown by attaching new nodes each with m
        # edges that are preferentially attached to existing nodes with
        # high degree.
        G = nx.barabasi_albert_graph(n, m, seed)
    elif name == 'powerlaw':
        # Grows graphs with powerlaw degree distribution and approximate
        # average clustering
        G = nx.powerlaw_cluster_graph(n, m, p, seed)
    elif name == 'regular':
        # Returns a random d-regular graph on n nodes
        G = nx.random_regular_graph(d, n, seed)
    else:
        print('Error, name not recognised or parameters meaningless')
    # Add weight to edges
    if name != 'toy':
        N = len(list(G.edges))
        capacities = np.random.randint(low=1, 
                high=70, 
                size=N).tolist()
        for edge, pos in zip(G.edges(data=True), range(N)):
            edge[2]['weight'] = capacities[pos]
    return G

def networkx_to_lists(G):
    '''
        Let G be a networkx graph.
         Define three parallel arrays:
            start, end, capacities
         such that edge "i" is (start[i], end[i])
    '''
    def decouple_edges(edges_list):
        start = [x[0] for x in edges_list]
        end = [x[1] for x in edges_list]
        capacities = [x[2]['weight'] for x in edges_list]
        return start, end, capacities
    edges = list(G.edges(data=True))
    start, end, capacities = decouple_edges(edges)
    return start, end, capacities

def build_maxflow_graph(start, end, capacities):
    # Instantiate a SimpleMaxFlow solver.
    maxflow = pywrapgraph.SimpleMaxFlow()
    # Add each arc.
    for i in range(0, len(start)):
        maxflow.AddArcWithCapacity(start[i], end[i], capacities[i])
    return maxflow 

def solve_max_flow(maxflow, source_node, sink_node, verbose=False):
    optimal_flow = None
    if maxflow.Solve(source_node, sink_node) == maxflow.OPTIMAL:
        optimal_flow = maxflow.OptimalFlow()
        num_arcs = maxflow.NumArcs()
        if verbose:
            print('Max flow:', optimal_flow)
            print('  Arc    Flow / Capacity')
            for i in range(num_arcs):
                print('%1s -> %1s   %3s  / %3s' % (
                    maxflow.Tail(i),
                    maxflow.Head(i),
                    maxflow.Flow(i),
                    maxflow.Capacity(i)))
            print('Source side min-cut:', maxflow.GetSourceSideMinCut())
            print('Sink side min-cut:', maxflow.GetSinkSideMinCut())
    else:
        print('There was an issue with the max flow input.')
    return optimal_flow

def generate_LP_data(G, s, t, capacities):
    """
        G is a networkx object
        s: source
        t: sink

        Let x(u,v) be the varible corresponding to
        edge e=(u,v) \in E. Problem is:

        max sum_{v : (s,v) in E} f(s,v)
        s.t.
            sum_{u: (u,v) in E} f(u,v) = sum_{w: (v,w) in E} f(v,w)
                forall v \in V - {s, t}
            f(u, v) <= q(u, v)
                forall (u,v) in E
            f(u, v) >= 0
                forall (u,v) in E
        
        We want to transform this to

        min c*x
        s.t.
            A_ub*x <= b_ub
            A_eq*x <= b_eq
    """
    # defines the natural ordering in the edges
    nodes = list(G.nodes)
    edges = list(G.edges)
    M     = len(nodes) 
    N     = len(edges)

    # in_edges[v]  = [(u, v) for u in V]
    # out_edges[v] = [(v, w) for w in V] 
    in_edges  = {}
    out_edges = {}
    for node in nodes:
        in_edges[node]  = []
        out_edges[node] = []
    for pos in range(N):
        i, j = edges[pos]
        in_edges[j].append(pos)
        out_edges[i].append(pos)

    # c
    c = np.zeros(N)
    c[np.array(out_edges[s])] = 1

    # A_eq
    A_eq = np.zeros((M,N))
    for row in range(M):
        node = nodes[row]
        assert( node == row )
        if node != s and node != t:
            A_eq[row, np.array(in_edges[node])] = 1 
            if out_edges[node]:
                assert(A_eq[row, np.array(out_edges[node])].sum() == 0)
                A_eq[row, np.array(out_edges[node])] = -1
    A_eq = np.delete(A_eq, (s,t), axis=0)
            
    # b_eq
    b_eq = np.zeros((M,1))
    b_eq = np.delete(b_eq, (s,t), axis=0)

    # A_ub 
    A_ub = None

    # b_ub
    b_ub = None

    # bounds
    bounds = [(0, c) for c in capacities]

    return (c, A_eq, b_eq, A_ub, b_ub, bounds)

def main():
    """MaxFlow simple interface example."""
    verbose = False

    # -------------
    # Generate data
    # -------------

    name = 'toy_2'
    G = generate_networkx_graph(name)
    start, end, capacities = networkx_to_lists(G)

    # -------------
    # Find the maximum flow between s and t
    # -------------

    s = 0 # source node
    t = 4 # sink node
    
    # Create maxflow object and solve with ortools
    maxflow = build_maxflow_graph(start, end, capacities)
    optimal_flow = solve_max_flow(maxflow, s, t)
    print('Max flow:', optimal_flow)

    # Generate LP data and solve with linprog
    #  to convert csc_matrx to array call .toarray()
    c, A_eq, b_eq, A_ub, b_ub, bounds = generate_LP_data(G, s, t, capacities)
    linprog_flow = linprog(-c, 
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds)
    print('Max flow (linprog):')
    print(linprog_flow)
    print(linprog_flow.x)

if __name__ == '__main__':
    main()
