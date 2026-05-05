"""
2026 zowie128

SICP (SUSCEPTIBLE-INFECTED with CONTACT PROCESS) diffusion model runner for hypergraphs. 

SICP is a diffusion model that simulates the spread of an infection (or information) through a hypergraph.
- Each infected node at time t chooses one of its hyperedges at random and tries to infect all other nodes in that hyperedge with probability beta.
- Infected nodes do not recover, and the process runs for T timesteps.

This script contains
- Loading in hypergraphs + communities from the data_loader
- Running SICP for a given hypergraph, seed set and parameters (beta, T) for multiple independent runs with controlled RNG seed
- Reporting per-run and aggregate statistics of the final infected counts
- Optionally, you can also use SICP to report the number of infections at each timestep. See the SICP function in methods/SICP.py.

"""
import random
import statistics

import numpy as np

from config import L_DATASETS, EL_DATASETS, seed_set_degree
from data_loader import (
    load_labeled_hypergraph,
    load_edge_labeled_hypergraph,
    load_labeled_hypergraph_cd,
    load_edge_labeled_hypergraph_cd,
)
from methods.SICP import SICP_set, SICP



def dataset_kind(dataset):
    if dataset in L_DATASETS:
        return "node"
    if dataset in EL_DATASETS or dataset in EL_DATASETS.values():
        return "edge"
    raise ValueError(f"Unknown dataset '{dataset}'. Available node datasets: {L_DATASETS}.\nAvailable edge datasets: {list(EL_DATASETS.keys())}")


def resolve_dataset_key(dataset):
    if dataset in EL_DATASETS:
        return EL_DATASETS[dataset]
    return dataset


def load_graph_and_communities(dataset, tau, verbose=True):
    kind = dataset_kind(dataset)
    key = resolve_dataset_key(dataset)

    if kind == "node":
        if tau is None:
            graph, clusters = load_labeled_hypergraph(key, source=True)
        else:
            graph, cd = load_labeled_hypergraph_cd(key, taus=(tau,), source=True, verbose=verbose)
            clusters = cd.get(tau, {})
    else:
        if tau is None:
            graph, clusters = load_edge_labeled_hypergraph(key, source=True)
        else:
            graph, cd = load_edge_labeled_hypergraph_cd(key, taus=(tau,), source=True, verbose=verbose)
            clusters = cd.get(tau, {})

    if verbose:
        print(f"Loaded dataset '{dataset}' ({kind}-labeled) with {len(graph)} hyperedges and {len(clusters)} communities")
        comm_sizes = sorted([len(c) for c in clusters.values()])
        if comm_sizes:
            print(f"  communities sizes: min={min(comm_sizes)}, max={max(comm_sizes)}, avg={statistics.mean(comm_sizes):.3f}")
    return graph, clusters


def main():

    # All datasets available, you can add your own to the data_loader.py
    DATASET_LIST = ['Algebra', 'Geometry', 'Music-Rev', 'Restaurants-Rev', 'Bars-Rev', 'contact-high-school','contact-primary-school']

    dataset = DATASET_LIST[1]  # Choose your dataset to run
    BETA = 0.3               # Infection probability
    T = 25                     # Timesteps to run SICP for
    RUNS = 25                  # Number of independent runs
    rng_seed = 79              # reproducible outcomes

    # Seed nodes to start infection from.  You can use the seed_selection.py script to generate seed sets with different methods and use them here. 
    # Can be added to config.py for easier access across scripts.
    # Be aware that seed nodes are dataset-specific, so make sure to use a seed set that corresponds to the dataset you choose to run SICP on.
    seed_set = seed_set_degree  #['102', '16', '13', '77', '8']      


    print("=== SICP runner ===")
    print(f"dataset={dataset}, beta={BETA}, T={T}, runs={RUNS}, seeds={sorted(seed_set)}, rng_seed={rng_seed}")

    # Load graph and communities
    # You can use communities for analysis, but SICP does not require it as input
    graph, communities = load_graph_and_communities(dataset, tau=2, verbose=True)

    if communities:
        print(f"\n=== community data loaded ({len(communities)} communities) ===")
        comm_sizes = sorted([len(c) for c in communities.values()])
        print(f"community sizes: {comm_sizes}")
    else:
        print("\nNo communities found for this dataset/tau combination")


    print("\n=== running SICP ===")
    print(f"seed set: {seed_set}")
    # Run SICP multiple times with a controlled RNG seed per run
    infected_results = []
    infected_node_sets = []

    for run in range(1, RUNS + 1):
        random.seed(rng_seed + run)
        infected_nodes = SICP_set(graph, seed_set, BETA, T)
        infected_node_sets.append(infected_nodes)
        print(f"Run {run}: {len(infected_nodes)} infected nodes")
        infected_results.append(len(infected_nodes))

    print("\n=== Aggregate summary ===")
    print(f"final count min={min(infected_results)}, max={max(infected_results)}, mean={statistics.mean(infected_results):.3f}, std={statistics.pstdev(infected_results):.3f}")
    # You can also use SICP to get the full series of infections over time, but here we just report the final count per run.
    # see the SICP function in methods/SICP.py for how to get the series

    return infected_results

def node_removal():
        # All datasets available, you can add your own to the data_loader.py
    DATASET_LIST = ['Algebra', 'Geometry', 'contact-primary-school']

    dataset_results = {}

    BETA = 0.3               # Infection probability
    T = 25                     # Timesteps to run SICP for
    RUNS = 10                  # Number of independent runs
    rng_seed = 175              # reproducible outcomes

    for dataset in DATASET_LIST:
        print(f"dataset ={dataset}, beta={BETA}, T={T}, runs={RUNS}, rng_seed={rng_seed}")
        graph, communities = load_graph_and_communities(dataset, tau=2, verbose=True)
        if communities:
            print(f"\n=== community data loaded ({len(communities)} communities) ===")
            comm_sizes = sorted([len(c) for c in communities.values()])
            print(f"community sizes: {comm_sizes}")
        else:
            print("\nNo communities found for this dataset/tau combination")
        
        node_ids = set.union(*communities.values())
        
        results_per_seed = [len(node_ids)]

        for single_seed in node_ids:

            # Seed nodes to start infection from.  You can use the seed_selection.py script to generate seed sets with different methods and use them here. 
            # Can be added to config.py for easier access across scripts.
            # Be aware that seed nodes are dataset-specific, so make sure to use a seed set that corresponds to the dataset you choose to run SICP on.
            seed_set = [single_seed]
            print(f"seed set: {seed_set}")

            # Run SICP multiple times with a controlled RNG seed per run
            infected_results = []
            infected_node_sets = []

            for run in range(1, RUNS + 1):
                random.seed(rng_seed + run)
                infected_nodes = SICP_set(graph, seed_set, BETA, T)
                infected_node_sets.append(infected_nodes)
                infected_results.append(len(infected_nodes))

            print(f"final count min={min(infected_results)}, max={max(infected_results)}, mean={statistics.mean(infected_results):.3f}, std={statistics.pstdev(infected_results):.3f}")
            # You can also use SICP to get the full series of infections over time, but here we just report the final count per run.
            # see the SICP function in methods/SICP.py for how to get the series

            dataset_results[dataset] = infected_results
    return dataset_results

if __name__ == "__main__":
    # main()
    node_removal()
