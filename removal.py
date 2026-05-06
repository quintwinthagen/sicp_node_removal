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

def community_based_removal(graph, communities, K, alpha=1.0, beta=0.15, strict_missing=True):
    """
    Community-aware node removal without Gini impurity.

    Score per node = alpha * bridge_score + beta * entropy_score

    bridge_score:
      Sum over incident inter-community hyperedges:
      spread_factor(edge) * boundary_participation(node, edge)

      spread_factor(edge) = (num_communities_in_edge - 1) / max(1, edge_size - 1)
      boundary_participation(node, edge) = 1 - p(node_community | edge)

    entropy_score:
      Entropy over community exposure in mixed incident hyperedges.
      Encourages nodes that connect many different communities.
    """
    inv_comm = inverse_communities(communities)

    node_bridge_score = defaultdict(float)
    node_exposure = defaultdict(Counter)  # node -> Counter(community -> weight)

    for hedge, nodes in graph.items():
        if not nodes:
            continue

        # Resolve node -> community for this edge
        node_to_c = {}
        missing_nodes = []
        for node in nodes:
            c = inv_comm.get(node, "UNK")
            if c == "UNK":
                missing_nodes.append(node)
            node_to_c[node] = c

        if missing_nodes:
            if strict_missing:
                raise ValueError(f"node(s) without recorded community in edge {hedge}: {missing_nodes[:5]}")
            # Skip this edge when community assignments are incomplete
            continue

        comm_counts = Counter(node_to_c.values())
        num_comms = len(comm_counts)

        # Only inter-community edges contribute
        if num_comms <= 1:
            continue

        edge_size = len(nodes)
        # Your spread factor idea, fixed: max(...) not math.max(...)
        spread_factor = (num_comms - 1) / max(1, edge_size - 1)

        # Node contributions in this edge
        for node in nodes:
            c_node = node_to_c[node]
            p_node_comm = comm_counts[c_node] / edge_size
            boundary_participation = 1.0 - p_node_comm

            contrib = spread_factor * boundary_participation
            node_bridge_score[node] += contrib

            # Exposure profile for entropy term
            for c, cnt in comm_counts.items():
                node_exposure[node][c] += cnt / edge_size

    # Final score with entropy regularizer
    node_final_score = {}
    for node, bscore in node_bridge_score.items():
        exposure_counter = node_exposure[node]
        z = sum(exposure_counter.values())

        if z <= 0:
            entropy = 0.0
        else:
            probs = [v / z for v in exposure_counter.values()]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        node_final_score[node] = alpha * bscore + beta * entropy

    # Deterministic ranking: score desc, then node id asc
    ranked_nodes = sorted(node_final_score.keys(), key=lambda n: (-node_final_score[n], str(n)))
    return ranked_nodes[:K]


def community_based_removal1(graph, communities, K):
    inv_comm = inverse_communities(communities)
    inter_comm_hedges = set()
    for hedge, nodes in graph.items():
        unique_communities = set()
        unique_communities.update([inv_comm.get(node, "UNK") for node in nodes])
        if "UNK" in unique_communities: raise ValueError("node has no recorded community")
        if len(unique_communities) > 1:
            inter_comm_hedges.add(hedge)


    node_to_inter_comm_occurences = Counter()
    for ic_hedge in inter_comm_hedges:
        node_to_inter_comm_occurences.update(graph[ic_hedge])

    return {el[0] for el in node_to_inter_comm_occurences.most_common(K)} #can cause differrences: equal-count ordering depends in insertion order

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