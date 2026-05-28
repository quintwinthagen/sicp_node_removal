""" 
    EPT: Equivalent Probability Transformation
    Transforms a hypergraph to a directed weighted graph, where weights are infection probability 

"""


from collections import defaultdict
from core.data_loader import Hypergraph

def compute_hyperdegree(graph: Hypergraph):
    """graph: dict[hedge_id -> iterable(nodes)] -> dict[node -> hyperdegree]"""
    hdeg = defaultdict(int)

    for _, nodes in graph.hyperedges.items():
        for u in nodes:
            hdeg[u] += 1
    return dict(hdeg)

def build_ept(graph: Hypergraph, beta=1.0):
    """
    Build directed weights w[u][v] = beta * (#shared hyperedges of u,v) / hdeg(u).

    Isolated nodes are not part of the EPT, since they have no transmission potential

    graph: dict[hedge_id -> iterable(nodes)]
    returns: dict[u -> dict[v -> weight]]
    """

    hdeg = compute_hyperdegree(graph)
    w = defaultdict(lambda: defaultdict(float))

    # Two-pass logic: hdeg known, now accumulate contributions per shared hyperedge.
    for _, nodes_iter in graph.hyperedges.items():
        nodes = list(nodes_iter)
        # For each u, this hyperedge contributes beta/hdeg(u) to every v in the same hyperedge.
        for u in nodes:
            du = hdeg.get(u, 0)
            if du == 0:
                continue
            contrib = beta / du
            for v in nodes:
                if v != u:
                    w[u][v] += contrib


    # Convert to normal dicts
    return {u: dict(nbrs) for u, nbrs in w.items()}