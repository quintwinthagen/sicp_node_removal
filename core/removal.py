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


## dict[hedge id --> {node ids}]
# def inverse_graph(graph):
#     node_to_hedge = defaultdict(list)
#     for hedge, nodes in graph.items():
#         for node in nodes:
#             node_to_hedge[node].append(hedge)
#     return node_to_hedge

# Use make_greedy_removal
## remove_nodes_greedily(count_ic_edges, (graph, communities), 14, 1) : returns graph with nodes removed greedily
# def remove_nodes_greedily(func: Callable[..., List[str]], args, K, batch):
#     full_iterations = K // batch
#     remaining_nodes = K % batch

#     g = args[0]
#     c = None

#     # Detect a communities mapping robustly (dict, defaultdict, etc.)
#     if len(args) > 1 and isinstance(args[1], Mapping):
#         c = args[1]
    

#     Ks = full_iterations * [batch]
#     if remaining_nodes:
#         Ks.append(remaining_nodes)

#     for i in range(full_iterations + (1 if remaining_nodes > 0 else 0)):
#         removal_amount = Ks[i]
#         if c is not None: to_remove = func(g, c, *args[2:], removal_amount)
#         else: to_remove = func(g, *args[1:], removal_amount)
#         g = remove_nodes(g, to_remove)

#         # Keep updating communities as long as a mapping was provided
#         if c is not None:
#             new_c = {}
#             for comm_label, nodes in c.items():
#                 pruned_nodes = set(nodes)
#                 for n in to_remove:
#                     pruned_nodes.discard(n)
#                 new_c[comm_label] = pruned_nodes
#             c = new_c

#     return g

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


def ept_bridge_out_strength_removal(graph: Hypergraph, communities, beta, K):
    bridge_out, _, _ = _ept_bridge_strengths(graph, communities, beta)
    return sorted(
        graph.nodes,
        key=lambda u: (bridge_out[u], str(u)),
        reverse=True
    )[:K]


def ept_bridge_in_strength_removal(graph: Hypergraph, communities, beta, K):
    _, bridge_in, _ = _ept_bridge_strengths(graph, communities, beta)
    return sorted(
        graph.nodes,
        key=lambda v: (bridge_in[v], str(v)),
        reverse=True
    )[:K]


def ept_bridge_in_and_out_total(graph, communities, beta, K):
    ept = build_ept(graph, beta)
    _, bridge_in, _ = _ept_bridge_strengths(graph, communities, beta)

    out_s = defaultdict(float)
    in_s  = defaultdict(float)

    for u, nbrs in ept.items():
        for v, w in nbrs.items():
            out_s[u] += w
            in_s[v]  += w

    score = {u: out_s[u] + in_s[u] + 2 * bridge_in[u] for u in graph.nodes}
    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
        reverse=True
    )[:K]


def ept_boundary_strength_removal(graph: Hypergraph, communities, beta, K):
    bridge_out, bridge_in, _ = _ept_bridge_strengths(graph, communities, beta)

    score = {x: bridge_out[x] + bridge_in[x] for x in graph.nodes}
    return sorted(
        graph.nodes,
        key=lambda x: (score[x], str(x)),
        reverse=True
    )[:K]



def ept_bridge_out_fraction_removal(graph: Hypergraph, communities, beta, K, eps=1e-12):
    bridge_out, _, out_strength = _ept_bridge_strengths(graph, communities, beta)

    score = {}
    for u in graph.nodes:
        score[u] = bridge_out[u] / (out_strength[u] + eps)

    return sorted(graph.nodes, key=lambda u: (score[u], str(u)), reverse=True)[:K]


def ept_bridge_broker_removal(graph: Hypergraph, communities, beta, K, eps=1e-12):
    bridge_out, bridge_in, _ = _ept_bridge_strengths(graph, communities, beta)

    score = {x: (bridge_in[x] + eps) * (bridge_out[x] + eps) for x in graph.nodes}
    return sorted(
        graph.nodes,
        key=lambda x: (score[x], str(x)),
        reverse=True
    )[:K]



def ept_boundary_ratio_removal(graph: Hypergraph, communities, beta, K, eps=1e-12):
    inv_comm = inverse_communities(communities)
    ept = build_ept(graph, beta)

    bridge_out = defaultdict(float)
    within_out = defaultdict(float)

    for u, nbrs in ept.items():
        cu = inv_comm.get(u, None)
        if cu is None:
            continue
        for v, w_uv in nbrs.items():
            cv = inv_comm.get(v, None)
            if cv is None:
                continue
            if cu == cv:
                within_out[u] += w_uv
            else:
                bridge_out[u] += w_uv

    score = {u: bridge_out[u] / (within_out[u] + eps) for u in graph.nodes}

    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
        reverse=True
    )[:K]

def ept_broker_strength(graph, beta, K, eps=1e-12):
    ept = build_ept(graph, beta)

    out_s = defaultdict(float)
    in_s  = defaultdict(float)

    for u, nbrs in ept.items():
        for v, w in nbrs.items():
            out_s[u] += w
            in_s[v]  += w


    score = {u: (in_s[u] + eps) * (out_s[u] + eps) for u in graph.nodes}

    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
        reverse=True
    )[:K]



def ept_pagerank_removal(graph: Hypergraph, beta, K, d=0.85, iters=50):
    ept = build_ept(graph, beta)

    # Collect EPT-visible nodes
    nodes = set(ept.keys())
    for u, nbrs in ept.items():
        nodes.update(nbrs.keys())
    nodes = list(nodes)

    if not nodes:
        return sorted(graph.nodes, key=str, reverse=True)[:K]

    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # Normalize outgoing weights to probabilities
    out_sum = defaultdict(float)
    for u, nbrs in ept.items():
        out_sum[u] = sum(nbrs.values())

    # PageRank vector
    pr = [1.0 / n] * n
    base = (1.0 - d) / n

    for _ in range(iters):
        new = [base] * n

        for u, nbrs in ept.items():
            su = out_sum[u]
            if su <= 0:
                continue
            pu = pr[idx[u]]
            for v, w in nbrs.items():
                new[idx[v]] += d * pu * (w / su)

        pr = new

    score = {u: 0.0 for u in graph.nodes}
    for u in nodes:
        score[u] = pr[idx[u]]

    return sorted(
        graph.nodes,
        key=lambda u: (score[u], str(u)),
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


def ept_out_strength(graph, beta, K):
    ept = build_ept(graph, beta)

    score = {
        u: sum(ept[u].values()) if u in ept else 0.0
        for u in graph.nodes
    }
    return sorted(
        score.keys(),
        key=lambda n: (score[n], str(n)),
        reverse=True,
    )[:K]


def count_communities_in_hedges_removal(graph: Hypergraph, communities, K):
    """Rank nodes by the number of unique communities seen across incident hyperedges."""
    inv_graph = inverse_graph_edges(graph)
    inv_comm = inverse_communities(communities)
    # Precompute each hyperedge's labeled communities once to avoid repeated scans.
    hedge_to_communities = {}
    for hedge, nodes in graph.hyperedges.items():
        hedge_to_communities[hedge] = {
            inv_comm[nei] for nei in nodes if nei in inv_comm
        }

    # Per node: count unique communities appearing across incident hyperedges.
    community_factor = {}
    for node, hedges in inv_graph.items():
        unique_communities = set()
        for hedge in hedges:
            unique_communities.update(hedge_to_communities[hedge])
        community_factor[node] = len(unique_communities)

    return sorted(
        community_factor.keys(),
        key=lambda n: (community_factor[n], str(n)),
        reverse=True,
    )[:K]

# def ic_hedges_to_hdeg(graph, communities, K):
#     """Rank nodes by inter-community incident edge count divided by hyperdegree."""
#     inv_comm = inverse_communities(communities)
#     inv_graph = inverse_graph(graph)
#     inter_comm_hedges = set()
#     for hedge, nodes in graph.items():
#         unique_communities = set()
#         unique_communities.update([inv_comm.get(node, "UNK") for node in nodes])
#         if "UNK" in unique_communities:
#             continue
#             # raise ValueError("node has no recorded community")
#         if len(unique_communities) > 1:
#             inter_comm_hedges.add(hedge)

#     node_to_inter_comm_occurences = Counter()
#     for ic_hedge in inter_comm_hedges:
#         node_to_inter_comm_occurences.update(graph[ic_hedge])
    
#     node_to_ic_hdeg_fraction = {}
#     for node, ic_count in node_to_inter_comm_occurences.items():
#         node_to_ic_hdeg_fraction[node] = ic_count / len(inv_graph[node]) # ic count / hdeg

#     return sorted(node_to_ic_hdeg_fraction.keys(), key=lambda n: node_to_ic_hdeg_fraction[n], reverse=True)[:K]



def count_ic_hedges(graph: Hypergraph, communities, K):
    """Select nodes that appear most often in inter-community hyperedges."""
    inv_comm = inverse_communities(communities)
    inter_comm_hedges = set()

    for hedge, nodes in graph.hyperedges.items():
        unique_communities = set()
        unique_communities.update([inv_comm.get(node, "UNK") for node in nodes])
        if "UNK" in unique_communities:
            continue
        if len(unique_communities) > 1:
            inter_comm_hedges.add(hedge)

    node_to_inter_comm_occurences = Counter()
    for ic_hedge in inter_comm_hedges:
        node_to_inter_comm_occurences.update(graph.hyperedges[ic_hedge])

    score = {
        node: node_to_inter_comm_occurences[node]
        for node in graph.nodes
    }

    return sorted(
        graph.nodes,
        key=lambda node: (score[node], str(node)),
        reverse=True
    )[:K]


def responsibility_weighted_hdeg_removal(graph: Hypergraph, communities, K):
    inv_comm = inverse_communities(communities)
    inv_graph = inverse_graph_edges(graph)

    hedge_to_communities = {}
    for hedge, nodes in graph.hyperedges.items():
        hedge_to_communities[hedge] = {
            inv_comm[n] for n in nodes if n in inv_comm
        }

    resp = defaultdict(int)

    # add responsibility score
    for node, hedges in inv_graph.items():
        node_comm = inv_comm[node]
        for hedge in hedges:
            before = hedge_to_communities[hedge]
            if node_comm not in before:
                continue

            after = before - {node_comm}
            resp[node] += max(0, len(before) - len(after))
    
    # add hyperdegree term
    for node, hedges in inv_graph.items():
        resp[node] += len(hedges)

    return sorted(
        graph.nodes,
        key=lambda node: (resp[node], str(node)),
        reverse=True
    )[:K]
    
# def filtered_hyperdegree_removal(graph: Hypergraph, communities, K):
#     """
#     Filtered hyperdegree: count only hyperedges for which the node is responsible for inter-community mixing.
#     """
#     inv_comm = inverse_communities(communities)
#     inv_graph = inverse_graph_edges(graph)

#     # Precompute community sets per hyperedge
#     hedge_to_communities = {}
#     for hedge, nodes in graph.hyperedges.items():
#         hedge_to_communities[hedge] = {
#             inv_comm[n] for n in nodes if n in inv_comm
#         }

#     score = defaultdict(int)

#     for node, hedges in inv_graph.items():
#         node_comm = inv_comm[node]
#         for hedge in hedges:
#             before = hedge_to_communities[hedge]
#             if node_comm not in before:
#                 continue

#             after = before - {node_comm}
#             # count this hyperedge only if node is responsible
#             if len(after) < len(before):
#                 score[node] += 1

#     return sorted(score, key=score.get, reverse=True)[:K]


def avg_hyperedge_size_removal(graph: Hypergraph, K: int):
    """Rank all nodes by average size of their incident hyperedges.
    Isolated nodes are assigned score 0.0 and therefore rank last.
    """
    incident_sizes = defaultdict(list)

    for nodes in graph.hyperedges.values():
        edge_size = len(nodes)
        for n in nodes:
            incident_sizes[n].append(edge_size)

    avg = {}
    for node in graph.nodes:
        sizes = incident_sizes.get(node)
        avg[node] = (sum(sizes) / len(sizes)) if sizes else 0.0

    ranking = sorted(avg, key=lambda node: (-avg[node], str(node)))
    return ranking[:K]

    

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

def random_based_removal(graph:Hypergraph, K):
    """Select K nodes uniformly at random from all nodes."""
    return random.sample(list(graph.nodes), K)


def remove_nodes(graph: Hypergraph, node_ids: list[str]):
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