from collections import Counter, defaultdict
from collections.abc import Mapping
import random

from core.data_loader import Hypergraph

from .ept import build_ept

def inverse_communities(communities):
    node_to_comm = {}
    for label, nodes in communities.items():
        for node in nodes:
            node_to_comm[node] = label
    return node_to_comm

def inverse_graph_edges(graph: Hypergraph):
    """
    Return node -> list of incident hyperedges for all nodes in the hypergraph.

    Isolated nodes are included and mapped to an empty list.
    """
    node_to_hedge = {node: [] for node in graph.nodes}

    for hedge, nodes in graph.hyperedges.items():
        for node in nodes:
            node_to_hedge[node].append(hedge)

    return node_to_hedge


def _ept_bridge_strengths(graph, communities, beta):
    """
    Returns:
      bridge_out[u] = sum_{v: c(v)!=c(u)} W[u][v]
      bridge_in[v]  = sum_{u: c(u)!=c(v)} W[u][v]
      out_strength[u] = sum_v W[u][v]
    Skips nodes not found in inv_comm (treated as UNK).
    """
    inv_comm = inverse_communities(communities)
    ept = build_ept(graph, beta)

    bridge_out = defaultdict(float)
    bridge_in  = defaultdict(float)
    out_strength = defaultdict(float)

    for u, nbrs in ept.items():
        cu = inv_comm.get(u, None)
        if cu is None:
            continue
        for v, w_uv in nbrs.items():
            cv = inv_comm.get(v, None)
            if cv is None:
                continue
            out_strength[u] += w_uv
            if cu != cv:
                bridge_out[u] += w_uv
                bridge_in[v]  += w_uv

    return bridge_out, bridge_in, out_strength


def ept_bridge_in_strength_removal(graph: Hypergraph, communities, beta, K):
    _, bridge_in, _ = _ept_bridge_strengths(graph, communities, beta)
    return sorted(
        graph.nodes,
        key=lambda v: (bridge_in[v], str(v)),
        reverse=True
    )[:K]


def ept_bridge_out_strength_removal(graph: Hypergraph, communities, beta, K):
    bridge_out, _, _ = _ept_bridge_strengths(graph, communities, beta)
    return sorted(
        graph.nodes,
        key=lambda v: (bridge_out[v], str(v)),
        reverse=True
    )[:K]

def ept_total_strength(graph, beta, K):
    ept = build_ept(graph, beta)

    out_s = defaultdict(float)
    in_s  = defaultdict(float)

    for u, nbrs in ept.items():
        for v, w in nbrs.items():
            out_s[u] += w
            in_s[v]  += w

    score = {u: out_s[u] + in_s[u] for u in graph.nodes}
    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
        reverse=True
    )[:K]


def ept_in_strength(graph, beta, K):
    ept = build_ept(graph, beta)

    in_s  = defaultdict(float)

    for u, nbrs in ept.items():
        for v, w in nbrs.items():
            in_s[v]  += w

    score = {u: in_s[u] for u in graph.nodes}
    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
        reverse=True
    )[:K]


def hyperdegree_based_removal(graph: Hypergraph, K):
    """Select nodes with highest hyperdegree (number of incident hyperedges)."""
    node_hyperdegrees = {node: 0 for node in graph.nodes}

    for hedge_nodes in graph.hyperedges.values():
        for node in hedge_nodes:
            node_hyperdegrees[node] += 1

    return sorted(
        graph.nodes,
        key=lambda n: (node_hyperdegrees[n], str(n)),
        reverse=True
    )[:K]


def degree_based_removal(graph: Hypergraph, K):
    """Select nodes with highest projected pairwise degree."""
    neighbors = {node: set() for node in graph.nodes}

    for hedge_nodes in graph.hyperedges.values():
        for node in hedge_nodes:
            neighbors[node].update(hedge_nodes - {node})

    node_degrees = {node: len(neighbors[node]) for node in graph.nodes}

    return sorted(
        graph.nodes,
        key=lambda n: (node_degrees[n], str(n)),
        reverse=True
    )[:K]


def average_hyperedge_size_removal(graph: Hypergraph, K):
    """
    Select nodes with the highest average size of incident hyperedges.
    Computes sum(|e|) / hdeg(v) for each node v.
    """
    node_sum = {node: 0 for node in graph.nodes}
    node_hdeg = {node: 0 for node in graph.nodes}

    for hedge_nodes in graph.hyperedges.values():
        size = len(hedge_nodes)
        for node in hedge_nodes:
            node_sum[node] += size
            node_hdeg[node] += 1

    score = {}
    for node in graph.nodes:
        if node_hdeg[node] > 0:
            score[node] = node_sum[node] / node_hdeg[node]
        else:
            score[node] = 0

    return sorted(
        graph.nodes,
        key=lambda n: (score[n], str(n)),
        reverse=True
    )[:K]

def random_based_removal(graph:Hypergraph, K):
    """Select K nodes uniformly at random from all nodes."""
    return random.sample(list(graph.nodes), K)


def remove_nodes(graph: Hypergraph, node_ids: list[str]) -> Hypergraph:
    node_ids = set(node_ids)

    ret_nodes = graph.nodes - node_ids
    ret_hedges = {}

    for edge_id, nodes in graph.hyperedges.items():
        s = {node for node in nodes if node not in node_ids}
        if len(s) > 1: # 1-degree 
            ret_hedges[edge_id] = s

    return Hypergraph(nodes=ret_nodes, hyperedges=ret_hedges)

def make_adaptive_removal(removal_func, batch=1):
    def wrapped(graph, *args, K):
        if K <= 0:
            return []

        comm = args[0] if args and isinstance(args[0], Mapping) else None
        rest = args[1:] if comm is not None else args

        ks = [batch] * (K // batch) + ([K % batch] if K % batch else [])

        removed = []

        for k in ks:
            to_rm = removal_func(graph, *( (comm,) if comm is not None else () ), *rest, k)
            if not to_rm:
                break

            removed += to_rm
            graph = remove_nodes(graph, to_rm)

            if comm is not None:
                rm = set(to_rm)
                comm = {c: nodes - rm for c, nodes in comm.items()}

        return removed[:K]

    return wrapped