from collections import defaultdict
import random

def SICP(G, S, beta, T, return_series=False):
    """
    SICP on a hypergraph.
    - G: dict {edge_id: iterable_of_nodes}
    - S: iterable of seed nodes
    - beta: infection probability 
    - T: number of timesteps
    - return_series: if True, return list of infected counts per timestep (len T+1); else final count
    """
    raise NotImplementedError("function not in use")

    infected = set(S)
    series = [len(infected)]

    # node -> list of incident edges (uniform choice over these)
    node_to_edges = defaultdict(list)
    for eid, nodes in G.items():
        for u in nodes:
            node_to_edges[u].append(eid)

    for _ in range(T):
        cur = list(infected)           # snapshot ⇒ synchronous updates
        new_infected = set()

        for v in cur:
            edges = node_to_edges.get(v)
            if not edges:
                continue
            e = random.choice(edges)        # choose ONE incident hyperedge uniformly
            for u in G[e]:
                if u not in infected and random.random() < beta:
                    new_infected.add(u)

        if not new_infected:
            series.append(series[-1])
            break

        infected |= new_infected
        series.append(len(infected))

    return series if return_series else len(infected)

# MIE needs the series
# Greedy needs len(infected)

def SICP_set(G, S, beta, T):
    """
    SICP on a hypergraph.
    - G: dict {edge_id: iterable_of_nodes}
    - S: iterable of seed nodes
    - beta: infection probability 
    - T: number of timesteps
    """
    from collections import defaultdict
    import random


    infected = set(S)

    node_to_edges = defaultdict(list)
    for eid, nodes in G.items():
        # for u in sorted(nodes, key=str):
        for u in nodes:
            # node_to_edges[u].sort();
            node_to_edges[u].append(eid)

    for _ in range(T):
        cur = sorted(infected, key=str)
        # cur = list(infected)
        new_infected = set()

        for v in cur:
            edges = node_to_edges.get(v)
            if not edges:
                continue
            e = random.choice(edges)
            for u in sorted(G[e], key=str):
                if u not in infected and random.random() < beta:
                    new_infected.add(u)

        # if not new_infected: #remove?
        #     break

        infected |= new_infected

    return infected  # final set of nodes
