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


def community_based_removal(graph, communities, K):
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

def avg_hyperedge_size_removal(graph, K):
    
    c = defaultdict(list)
    for _, nodes in graph.items():
        for n in nodes:
            c[n].append(len(nodes))
    avg = {}
    for node, sizes in c.items():
        avg[node] = sum(sizes) / len(sizes)
    
    ranking = sorted(avg.keys(), key=lambda i: avg[i], reverse=True)
    return ranking[:K]

def hyperdegree_based_removal(graph, K):
    return hyperdegree(graph, K)

def degree_based_removal(graph, K):
    return degree(graph, K)   

def random_based_removal(graph, K):
    all_nodes = list(set.union(*graph.values()))
    return random.sample(all_nodes, K)

def remove_nodes(graph, node_ids):
    removed_graph = {}
    for edge_id, nodes in graph.items():
        s = {node for node in nodes if str(node) not in node_ids}
        if s:
            removed_graph[edge_id] = s
    
    return removed_graph