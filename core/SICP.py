from collections import defaultdict
import random

from core.data_loader import Hypergraph

def SICP(G: Hypergraph, S, beta, T, return_series=False):
    """
    SICP on a hypergraph.
    - G: Hypergraph
    - S: iterable of seed nodes
    - beta: infection probability 
    - T: number of timesteps
    - return_series: if True, return list of infected-node sets per timestep (len T+1); else final set
    """
    invalid = set(S) - G.nodes
    if invalid:
        raise ValueError(f"Seed nodes not in hypergraph: {invalid}")
    infected = set(S)
    series = [set(infected)] 

    node_to_edges = getattr(G, "_sicp_node_to_edges", None)
    sorted_hyperedges = getattr(G, "_sicp_sorted_hyperedges", None)

    if node_to_edges is None or sorted_hyperedges is None:
        node_to_edges = defaultdict(list)
        sorted_hyperedges = {}

        for eid, nodes in G.hyperedges.items():
            sorted_hyperedges[eid] = sorted(nodes, key=str)
            for u in nodes:
                node_to_edges[u].append(eid)

        G._sicp_node_to_edges = node_to_edges
        G._sicp_sorted_hyperedges = sorted_hyperedges

    for _ in range(T):
        cur = sorted(infected, key=str)
        new_infected = set()

        for v in cur:
            edges = node_to_edges.get(v)
            if not edges:
                # v may be isolated, and can thus not spread, but remains infected
                continue
            e = random.choice(edges)
            for u in sorted_hyperedges[e]:
                if u not in infected and random.random() < beta:
                    new_infected.add(u)

        # if not new_infected:
        #     series.append(series[-1])
        #     break

        infected |= new_infected

        if return_series:
            series.append(set(infected))

    return series if return_series else infected

# MIE needs the series
# Greedy needs len(infected)

def SICP_set(G: Hypergraph, S, beta, T) -> set[str]:
    """
    SICP on a hypergraph.
    - G: Hypergraph dataclass
    - S: iterable of seed nodes
    - beta: infection probability 
    - T: number of timesteps
    """

    invalid = set(S) - G.nodes
    if invalid:
        raise ValueError(f"Seed nodes not in hypergraph: {invalid}")
    infected = set(S)

    node_to_edges: dict[str, list[str]] = defaultdict(list)

    for eid, nodes in G.hyperedges.items():
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
                # v may be isolated, in that case it cannot spread in this step.
                continue
            e = random.choice(edges)
            for u in sorted(G.hyperedges[e], key=str):
                if u not in infected and random.random() < beta:
                    new_infected.add(u)

        # if not new_infected: #remove?
        #     break

        infected |= new_infected

    return infected  # final set of nodes
