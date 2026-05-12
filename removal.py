import math
from collections import Counter, defaultdict
import random

from methods.baselines import degree, hyperdegree

def inverse_communities(communities):
    node_to_comm = {}
    for label, nodes in communities.items():
        for node in nodes:
            node_to_comm[node] = label
    return node_to_comm

def inverse_graph(graph):
    node_to_hedge = defaultdict(list)
    for hedge, nodes in graph.items():
        for node in nodes:
            node_to_hedge[node].append(hedge)
    return node_to_hedge


def count_communities_in_hedges_removal(graph, communities, K):
    """Rank nodes by the number of unique communities seen across incident hyperedges."""
    inv_graph = inverse_graph(graph)
    inv_comm = inverse_communities(communities)
    # Precompute each hyperedge's labeled communities once to avoid repeated scans.
    hedge_to_communities = {}
    for hedge, nodes in graph.items():
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


def count_ic_hedges(graph, communities, K):
    """Select nodes that appear most often in inter-community hyperedges."""
    inv_comm = inverse_communities(communities)
    inter_comm_hedges = set()
    for hedge, nodes in graph.items():
        unique_communities = set()
        unique_communities.update([inv_comm.get(node, "UNK") for node in nodes])
        if "UNK" in unique_communities:
            continue
            # raise ValueError("node has no recorded community")
        if len(unique_communities) > 1:
            inter_comm_hedges.add(hedge)

    node_to_inter_comm_occurences = Counter()
    for ic_hedge in inter_comm_hedges:
        node_to_inter_comm_occurences.update(graph[ic_hedge])

    return {el[0] for el in node_to_inter_comm_occurences.most_common(K)} #can cause differrences: equal-count ordering depends in insertion order

def responsibility_weighted_hdeg_removal(graph, communities, K):

    inv_comm = inverse_communities(communities)
    inv_graph = inverse_graph(graph)

    hedge_to_communities = {}
    for hedge, nodes in graph.items():
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


    return sorted(resp, key=resp.get, reverse=True)[:K]
    
def filtered_hyperdegree_removal(graph, communities, K):
    """
    Filtered hyperdegree: count only hyperedges for which the node is responsible for inter-community mixing.
    """
    inv_comm = inverse_communities(communities)
    inv_graph = inverse_graph(graph)

    # Precompute community sets per hyperedge
    hedge_to_communities = {}
    for hedge, nodes in graph.items():
        hedge_to_communities[hedge] = {
            inv_comm[n] for n in nodes if n in inv_comm
        }

    score = defaultdict(int)

    for node, hedges in inv_graph.items():
        node_comm = inv_comm[node]
        for hedge in hedges:
            before = hedge_to_communities[hedge]
            if node_comm not in before:
                continue

            after = before - {node_comm}
            # count this hyperedge only if node is responsible
            if len(after) < len(before):
                score[node] += 1

    return sorted(score, key=score.get, reverse=True)[:K]




# def avg_hyperedge_size_removal(graph, K):
#     """Rank nodes by average size of their incident hyperedges."""
    
#     c = defaultdict(list)
#     for _, nodes in graph.items():
#         for n in nodes:
#             c[n].append(len(nodes))
#     avg = {}
#     for node, sizes in c.items():
#         avg[node] = sum(sizes) / len(sizes)
    
#     ranking = sorted(avg.keys(), key=lambda i: avg[i], reverse=True)
#     return ranking[:K]
    

def hyperdegree_based_removal(graph, K):
    """Select nodes with highest hyperdegree (number of incident hyperedges)."""
    return hyperdegree(graph, K)

def degree_based_removal(graph, K):
    """Select nodes with highest projected pairwise degree."""
    return degree(graph, K)   

def random_based_removal(graph, K):
    """Select K nodes uniformly at random from all nodes."""
    all_nodes = list(set.union(*graph.values()))
    return random.sample(all_nodes, K)

def remove_nodes(graph, node_ids):
    removed_graph = {}
    for edge_id, nodes in graph.items():
        s = {node for node in nodes if str(node) not in node_ids}
        if s:
            removed_graph[edge_id] = s
    
    return removed_graph