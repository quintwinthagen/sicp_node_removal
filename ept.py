""" 
    EPT: Equivalent Probability Transformation
    Transforms a hypergraph to a directed weighted graph, where weights are infection probability 

"""


from collections import defaultdict
import math
import heapq

def compute_hyperdegree(graph):
    """graph: dict[hedge_id -> iterable(nodes)] -> dict[node -> hyperdegree]"""
    hdeg = defaultdict(int)
    for _, nodes in graph.items():
        for u in nodes:
            hdeg[u] += 1
    return dict(hdeg)

def build_ept(graph, beta=1.0):
    """
    Build directed weights w[u][v] = beta * (#shared hyperedges of u,v) / hdeg(u).

    graph: dict[hedge_id -> iterable(nodes)]
    returns: dict[u -> dict[v -> weight]]
    """

    hdeg = compute_hyperdegree(graph)
    w = defaultdict(lambda: defaultdict(float))

    # Two-pass logic: hdeg known, now accumulate contributions per shared hyperedge.
    for _, nodes_iter in graph.items():
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