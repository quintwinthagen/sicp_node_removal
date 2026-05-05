import math

from baselines import degree, hyperdegree

def remove_by_degree(graph, p):
    N = len(set.union(*graph.values()))
    return degree(graph, math.floor(p * float(N)))   

def remove_nodes(graph, node_ids):
    removed_graph = {}
    for edge_id, nodes in graph.items():
        s = {node for node in nodes if str(node) not in node_ids}
        if s:
            removed_graph[edge_id] = s
    
    return removed_graph