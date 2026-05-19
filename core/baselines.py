from collections import defaultdict, Counter

def hyperdegree(G, K):
    node_hyperdegrees = defaultdict(int)
    for hedge_nodes in G.values():
        for node in hedge_nodes:
            node_hyperdegrees[node] += 1

    # Sort deterministically: by degree desc, then node id asc
    top_nodes = sorted(
        node_hyperdegrees.keys(),
        key=lambda n: (node_hyperdegrees[n], n),
        reverse=True
    )[:K]

    return top_nodes 


def degree(G, K):
    neighbors = defaultdict(set)
    for hedge_nodes in G.values():
        for node in hedge_nodes:
            neighbors[node].update(hedge_nodes - {node})

    node_degrees = {node: len(neighs) for node, neighs in neighbors.items()}

    top_nodes = sorted(
        node_degrees.keys(),
        key=lambda n: (node_degrees[n], n),
        reverse=True
    )[:K]

    return top_nodes  
