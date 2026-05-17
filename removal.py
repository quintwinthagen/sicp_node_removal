import math
from collections import Counter, defaultdict
import random

from methods.baselines import degree, hyperdegree
from ept import build_ept

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


def generalized_jaccard(sets_list):
    if not sets_list:
        return 0.0

    intersection = set.intersection(*sets_list)
    union = set.union(*sets_list)

    # Avoid division by zero
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def ept_out_strength(graph, beta, K):
    ept = build_ept(graph, beta)

    score = {v : (sum(weights.values())) for v, weights in ept.items()}
    return sorted(
        score.keys(),
        key=lambda n: (score[n], str(n)),
        reverse=True,
    )[:K]


def average_pairwise_jaccard(sets_list):
    if len(sets_list) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(sets_list)):
        for j in range(i + 1, len(sets_list)):
            a = sets_list[i]
            b = sets_list[j]
            union = a | b
            if not union:
                continue
            total += len(a & b) / len(union)
            count += 1

    return total / count if count else 0.0


def jaccard_overlap(graph, K):
    """Select connector-like nodes whose incident hyperedges have low overlap among neighbors."""
    score = {}
    inv_graph = inverse_graph(graph)

    for node, hedges in inv_graph.items():
        if len(hedges) <= 1:
            continue

        edge_sets = [set(graph[hedge]) - {node} for hedge in hedges]
        mean_similarity = average_pairwise_jaccard(edge_sets)
        score[node] = 1.0 - mean_similarity

    return sorted(
        score.keys(),
        key=lambda n: (score[n], str(n)),
        reverse=True,
    )[:K]


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


def PHG_core_boundary_strategy(graph, communities, K):
    inv_comm = inverse_communities(communities)
    inv_graph = inverse_graph(graph)

    # --- Degree (C_D) ---
    degs = defaultdict(int)
    for node, hedges in inv_graph.items():
        for hedge in hedges:
            degs[node] += len(graph.get(hedge, [])) - 1

    # --- Community sizes ---
    comm_sizes = {c: len(nodes) for c, nodes in communities.items()}

    # --- CN(v) and AvgNS(v) ---
    CN = {}
    AvgNS = {}

    for node, hedges in inv_graph.items():
        node_comm = inv_comm.get(node, "UNK")
        if node_comm == "UNK":
            continue

        neighbor_comms = set()

        for hedge in hedges:
            for neigh in graph.get(hedge, []):
                if neigh == node:
                    continue
                neigh_comm = inv_comm.get(neigh, "UNK")
                if neigh_comm == "UNK":
                    continue
                if neigh_comm != node_comm:
                    neighbor_comms.add(neigh_comm)

        CN[node] = len(neighbor_comms)

        if CN[node] == 0:
            AvgNS[node] = 0.0
        else:
            AvgNS[node] = sum(comm_sizes[c] for c in neighbor_comms) / CN[node]

    # --- CS(v): own community size ---
    CS = {node: comm_sizes.get(inv_comm.get(node), 0) for node in inv_graph.keys()}

    # --- Min-max normalization ---
    def normalize(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    degs_n = normalize(degs)
    CN_n = normalize(CN)
    AvgNS_n = normalize(AvgNS)
    CS_n = normalize(CS)

    # --- Boundary detection ---
    is_boundary = {node: (CN.get(node, 0) > 0) for node in inv_graph.keys()}

    # --- Community Influence (CI) ---
    CI = {}

    for node in inv_graph.keys():
        if is_boundary[node]:
            CI[node] = (
                degs_n.get(node, 0.0)
                + CN_n.get(node, 0.0)
                + AvgNS_n.get(node, 0.0)
            ) / 3.0
        else:
            CI[node] = (
                degs_n.get(node, 0.0)
                + CS_n.get(node, 0.0)
            ) / 2.0

    # --- Return top K ---
    return sorted(CI.keys(), key=CI.get, reverse=True)[:K]


def PHG_community_influence(graph, communities, K):
    # for boundary nodes (nodes that connect multiple communities):
    # (hyper)degree of node + number of connected communities + (average size of communities)/3

    inv_comm = inverse_communities(communities)

    # hyperdegree (not used currently)
    
    inv_graph = inverse_graph(graph)
    hdegs = { node: len(hedges) for node, hedges in inv_graph.items()}

    # degree

    degs = defaultdict(int)
    for node, hedges in inv_graph.items():
        for hedge in hedges:
            degs[node] += len(graph.get(hedge, []))-1

    
    # number of connected communities
    
    # AvgNs (average size of communities of neighbours)

    comm_sizes = {comm: len(nodes) for comm, nodes in communities.items()}

    connected_community_count = {} # CN(v)
    avg_neighbor_comm_size = {}  # AvgNS(v)

    for node, hedges in inv_graph.items():
        node_comm = inv_comm.get(node, "UNK")
        if node_comm == "UNK":
            continue

        neighbor_comms = set()

        for hedge in hedges:
            for neighbor in graph.get(hedge, []):
                if neighbor == node:
                    continue
                neigh_comm = inv_comm.get(neighbor, "UNK")
                if neigh_comm == "UNK":
                    continue
                if neigh_comm != node_comm:
                    neighbor_comms.add(neigh_comm)
        
        connected_community_count[node] = len(neighbor_comms)

        cn = len(neighbor_comms)
        if cn == 0:
            avg_neighbor_comm_size[node] = 0.0
        else:
            avg_neighbor_comm_size[node] = sum(comm_sizes[c] for c in neighbor_comms) / cn
    
    community_influence_measure = defaultdict(float)

    for node, measure in degs.items():
        community_influence_measure[node] += measure

    for node, measure in connected_community_count.items():
        community_influence_measure[node] += measure

    for node, measure in avg_neighbor_comm_size.items():
        community_influence_measure[node] += measure / 3.0

    return sorted(community_influence_measure.keys(), key=community_influence_measure.get, reverse=True)[:K]


def avg_hyperedge_size_removal(graph, K):
    """Rank nodes by average size of their incident hyperedges."""
    
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