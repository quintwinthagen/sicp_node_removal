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
from collections import defaultdict
import random
import statistics
import sys
import time

import numpy as np

from core.config import L_DATASETS, EL_DATASETS, seed_set_degree
from core.data_loader import (
    Hypergraph,
    load_labeled_hypergraph,
    load_edge_labeled_hypergraph,
    load_labeled_hypergraph_cd,
    load_edge_labeled_hypergraph_cd,
)
from core.SICP import SICP_set, SICP

from core.removal import *
# from removal import (
#     # avg_hyperedge_size_removal,
#     filtered_hyperdegree_removal,
#     PHG_core_boundary_strategy,
#     remove_nodes, 
#     degree_based_removal, 
#     hyperdegree_based_removal,
#     # count_ic_hedges, 
#     random_based_removal,
#     # count_communities_in_hedges_removal,
#     # ic_hedges_to_hdeg,
#     responsibility_weighted_hdeg_removal,
# )

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


def load_graph_and_communities_old_structure(dataset, tau, verbose=True, source=True):
    kind = dataset_kind(dataset)
    key = resolve_dataset_key(dataset)

    if kind == "node":
        if tau is None:
            graph, clusters = load_labeled_hypergraph(key, source=source)
        else:
            graph, cd = load_labeled_hypergraph_cd(key, taus=(tau,), source=source, verbose=verbose)
            clusters = cd.get(tau, {})
    else:
        if tau is None:
            graph, clusters = load_edge_labeled_hypergraph(key, source=source)
        else:
            graph, cd = load_edge_labeled_hypergraph_cd(key, taus=(tau,), source=source, verbose=verbose)
            clusters = cd.get(tau, {})

    if verbose:
        print(f"Loaded dataset '{dataset}' ({kind}-labeled) with {len(graph)} hyperedges and {len(clusters)} communities")
        comm_sizes = sorted([len(c) for c in clusters.values()])
        if comm_sizes:
            print(f"  communities sizes: min={min(comm_sizes)}, max={max(comm_sizes)}, avg={statistics.mean(comm_sizes):.3f}")
    return graph, clusters


## returns dict with: graph["edges"]: dict[int, set[str]] and graph["nodes"]: set[str]

def load_graph_and_communities(dataset, tau, verbose=True, source=True) -> tuple[Hypergraph, dict]:
    graph_o, comm_o = load_graph_and_communities_old_structure(dataset, tau, verbose=verbose, source=source)
    nodes = set.union(*graph_o.values())
    hg = Hypergraph(nodes, graph_o)

    return hg, comm_o

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
    # DATASET_LIST = ['contact-primary-school']
    # DATASET_LIST = ['contact-primary-school']
    DATASET_LIST = ['Geometry']
    # DATASET_LIST = ['Algebra', 'Geometry']

    BETA = 0.3               # Infection probability
    T = 25                     # Timesteps to run SICP for
    RUNS = 10                  # Number of independent runs
    rng_seed = 175              # reproducible outcomes
    p = 0.05                # node removal fraction
    seed_iterations = 20000

    for dataset in DATASET_LIST:
        print(f"dataset={dataset}, beta={BETA}, T={T}, runs={RUNS}, rng_seed={rng_seed}, p={p}")
        temp_graph, communities = load_graph_and_communities(dataset, tau=2, verbose=True)

        all_nodes = set.union(*temp_graph.values())
        N_ = len(all_nodes)
        K = int(p * float(N_))
        
        if communities:
            print(f"\n=== community data loaded ({len(communities)} communities) ===")
            comm_sizes = sorted([len(c) for c in communities.values()])
            print(f"community sizes: {comm_sizes}")
        else:
            print("\nNo communities found for this dataset/tau combination")
        
        # graph_comm = remove_nodes(temp_graph, count_ic_hedges(temp_graph, communities, K))
        # graph_ic_hdeg_fraction = remove_nodes(temp_graph, ic_hedges_to_hdeg(temp_graph, communities, K))
        # graph_count_comms = remove_nodes(temp_graph, count_communities_in_hedges_removal(temp_graph, communities, K))
        # graph_PHG_core_boundary_strategy = remove_nodes(temp_graph, PHG_core_boundary_strategy(temp_graph, communities, K))
        print(f"starting time: {time.asctime()}")
        a = 0
        # graph_jaccard_overlap = remove_nodes(temp_graph, jaccard_overlap(temp_graph, K))
        # print(f"time checkpoint { (a := a+1)}: {time.asctime()}")

        # graph_responsibility_weighted_hdeg = remove_nodes(temp_graph, responsibility_weighted_hdeg_removal(temp_graph, communities, K))
        # graph_filtered_hyperdegree = remove_nodes(temp_graph, filtered_hyperdegree_removal(temp_graph, communities, K))
        graph_ept_out_strength = remove_nodes(temp_graph, ept_out_strength(temp_graph, BETA, K))
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")
        graph_rand = remove_nodes(temp_graph, random_based_removal(temp_graph, K))
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")
        graph_hdeg = remove_nodes(temp_graph, hyperdegree_based_removal(temp_graph, K))
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")
        graph_deg  = remove_nodes(temp_graph, degree_based_removal(temp_graph, K))
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")


        # graph_avg_degs = remove_nodes(temp_graph, avg_hyperedge_size_removal(temp_graph, K))

        # oc = len(get_graph_nodes(temp_graph))
        # lc = len(get_graph_nodes(graph_comm))
        # rc = len(get_graph_nodes(graph_rand))
        # hc = len(get_graph_nodes(graph_hdeg))
        # dc = len(get_graph_nodes(graph_deg))
        # ac = len(get_graph_nodes(graph_avg_degs))
        # print(oc, lc, rc, hc, dc, ac)

        # rps_comm = run_configured_sicp(graph_comm, seed_iterations)
        # print(f"results_comm: mean infected ({statistics.mean(rps_comm.values())})")

        # rps_ic_hdeg_fraction = run_configured_sicp(graph_ic_hdeg_fraction, seed_iterations)
        # print(f"results_ic_hdeg_fraction: mean infected ({statistics.mean(rps_ic_hdeg_fraction.values())})")

        # rps_count_comms = run_configured_sicp(graph_count_comms, seed_iterations)
        # print(f"results_count_comms: mean infected ({statistics.mean(rps_count_comms.values())})")

        # rps_avg_degs = run_configured_sicp(graph_avg_degs, seed_iterations)
        # print(f"results_avg_degs: mean infected ({statistics.mean(rps_avg_degs.values())})")

        # rps_PHG_core_boundary_strategy = run_configured_sicp(graph_PHG_core_boundary_strategy, seed_iterations)
        # print(f"PHG_core_boundary_strategy: mean infected ({statistics.mean(rps_PHG_core_boundary_strategy.values())})")
        
        # rps_jaccard_overlap = run_configured_sicp(graph_jaccard_overlap, seed_iterations)
        # print(f"graph_jaccard_overlap: mean infected ({statistics.mean(rps_jaccard_overlap.values())})")
        # print(f"time checkpoint { (a := a+1)}: {time.asctime()}")

        # rps_graph_responsibility_weighted_hdeg = run_configured_sicp(graph_responsibility_weighted_hdeg, seed_iterations)
        # print(f"results_graph_responsibility_weighted_hdeg: mean infected ({statistics.mean(rps_graph_responsibility_weighted_hdeg.values())})")

        # rps_graph_filtered_hyperdegree = run_configured_sicp(graph_filtered_hyperdegree, seed_iterations)
        # print(f"results_graph_filtered_hyperdegree: mean infected ({statistics.mean(rps_graph_filtered_hyperdegree.values())})")

        rps_ept_out_strength = run_configured_sicp(graph_ept_out_strength, seed_iterations)
        print(f"results_ept_out_strength mean infected ({statistics.mean(rps_ept_out_strength.values())})")
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")

        rps_rand = run_configured_sicp(graph_rand, seed_iterations)
        print(f"results_rand: mean infected ({statistics.mean(rps_rand.values())})")
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")

        rps_hdeg = run_configured_sicp(graph_hdeg, seed_iterations)
        print(f"results_hdeg: mean infected ({statistics.mean(rps_hdeg.values())})")
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")

        rps_deg  = run_configured_sicp(graph_deg, seed_iterations)
        print(f"results_deg : mean infected ({statistics.mean(rps_deg.values() )})")
        print(f"time checkpoint { (a := a+1)}: {time.asctime()}")   

def run_configured_sicp(
    graph: Hypergraph,
    seed_iterations,
    beta=0.3,
    T=25,
    runs=10,
    rng_seed=175,
):
    raise NotImplementedError()
#     node_ids = graph.nodes
#     prevalences_per_timestep = defaultdict(set)
#     results_per_seed = {}
#     cnt = 0
#     for single_seed in node_ids:
#         cnt += 1
#         if cnt > seed_iterations: break

#         # print(type(single_seed))

#         # Seed nodes to start infection from.  You can use the seed_selection.py script to generate seed sets with different methods and use them here. 
#         # Can be added to config.py for easier access across scripts.
#         # Be aware that seed nodes are dataset-specific, so make sure to use a seed set that corresponds to the dataset you choose to run SICP on.
#         seed_set = [single_seed]
#         # print(f"seed set: {seed_set}")

#         # Run SICP multiple times with a controlled RNG seed per run
#         infected_results = []
#         infected_node_sets = []

#         for run in range(1, runs + 1):
#             random.seed(rng_seed + run)
#             infected_nodes = SICP_set(graph, seed_set, beta, T)
#             infected_node_sets.append(infected_nodes)
#             infected_results.append(len(infected_nodes))


#         # print(f"final count min={min(infected_results)}, max={max(infected_results)}, mean={statistics.mean(infected_results):.3f}, std={statistics.pstdev(infected_results):.3f}")
#         # You can also use SICP to get the full series of infections over time, but here we just report the final count per run.
#         # see the SICP function in methods/SICP.py for how to get the series

#         results_per_seed[single_seed] = statistics.mean(infected_results)
#     return results_per_seed


def run_configured_sicp_intermediates(
    graph: Hypergraph,
    seed_iterations,
    beta=0.02,
    T=25,
    runs=10,
    rng_seed=175,
):
    node_ids = graph.nodes
    prevalences_per_timestep = [[] for _ in range(T + 1)]
    cnt = 0
    for node_idx, single_seed in enumerate(node_ids):
        cnt += 1
        if seed_iterations and (cnt > seed_iterations): break

        # print(type(single_seed))

        # Seed nodes to start infection from.  You can use the seed_selection.py script to generate seed sets with different methods and use them here. 
        # Can be added to config.py for easier access across scripts.
        # Be aware that seed nodes are dataset-specific, so make sure to use a seed set that corresponds to the dataset you choose to run SICP on.
        seed_set = [single_seed]
        # print(f"seed set: {seed_set}")

        for run in range(1, runs + 1):
            unique_seed = rng_seed + (node_idx * 100000) + (run * 1000)
            random.seed(rng_seed)

            infected_nodes_series = SICP(graph, seed_set, beta, T, return_series=True)
            for idx, infected in enumerate(infected_nodes_series):
                prevalences_per_timestep[idx].append(len(infected))
            
        # print(f"final count min={min(infected_results)}, max={max(infected_results)}, mean={statistics.mean(infected_results):.3f}, std={statistics.pstdev(infected_results):.3f}")
        # You can also use SICP to get the full series of infections over time, but here we just report the final count per run.
        # see the SICP function in methods/SICP.py for how to get the series

    return [statistics.mean(step_values) if step_values else 0.0 for step_values in prevalences_per_timestep]


if __name__ == "__main__": 
    # main()
    node_removal()
