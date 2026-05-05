"""
2026 zowie128

Seed selection algorithm runner for hypergraphs.
This script allows you to select seed nodes from a hypergraph using simple heuristics such as degree or hyperdegree.
You can implement your own seed selection method here as well, and use the generated seed set to run SICP in main.py.
"""

import random
import statistics

from config import L_DATASETS, EL_DATASETS
from data_loader import (
    load_labeled_hypergraph,
    load_edge_labeled_hypergraph,
    load_labeled_hypergraph_cd,
    load_edge_labeled_hypergraph_cd,
)
from methods.baselines import degree, hyperdegree
from methods.SICP import SICP_set

def dataset_kind(dataset):
    if dataset in L_DATASETS:
        return "node"
    if dataset in EL_DATASETS or dataset in EL_DATASETS.values():
        return "edge"
    raise ValueError(f"Unknown dataset '{dataset}'. node: {L_DATASETS}; edge: {list(EL_DATASETS.keys())}")


def resolve_dataset_key(dataset):
    return EL_DATASETS.get(dataset, dataset)


def load_graph(dataset, tau=2, verbose=True):
    kind = dataset_kind(dataset)
    key = resolve_dataset_key(dataset)

    if kind == "node":
        if tau is None:
            graph, communities = load_labeled_hypergraph(key, source=True)
        else:
            graph, cd = load_labeled_hypergraph_cd(key, taus=(tau,), source=True, verbose=verbose)
            communities = cd.get(tau, {})
    else:
        if tau is None:
            graph, communities = load_edge_labeled_hypergraph(key, source=True)
        else:
            graph, cd = load_edge_labeled_hypergraph_cd(key, taus=(tau,), source=True, verbose=verbose)
            communities = cd.get(tau, {})

    if verbose:
        print(f"Loaded {dataset} ({kind}) with {len(graph)} hyperedges and {len(communities)} communities")
    return graph, communities


def select_seeds(graph, method, k):
    if method.lower() == "degree":
        seeds = degree(graph, k)
    elif method.lower() == "hyperdegree":
        seeds = hyperdegree(graph, k)
    else:
        raise ValueError(f"Method must be 'degree' or 'hyperdegree', got '{method}'")
    return seeds


def run_sicp_rounds(graph, seeds, beta, T, runs, base_seed):
    final_sets = []

    for r in range(1, runs + 1):
        random.seed(base_seed + r)
        infected = SICP_set(graph, seeds, beta, T)
        final_sets.append(set(map(str, infected)))

    counts = [len(s) for s in final_sets]
    print("\nSICP replications:")
    for r, c in enumerate(counts, start=1):
        print(f"  run {r}: {c} infected")

    print("\nSummary:")
    print(f"  min={min(counts)}, max={max(counts)}, mean={statistics.mean(counts):.2f}, std={statistics.pstdev(counts):.2f}")

    return final_sets


def main():
    """
    You can use the generated seed set from this script to run SICP in main.py.
    Edit the constants below to choose dataset and method for which to run SICP
    If you implement your own method, that needs simulation of SICP to evaluate, you must add extra parameters such as BETA, T and RUNS.
    """
    DATASET_LIST = ['Algebra', 'Geometry', 'Music-Rev', 'Restaurants-Rev', 'Bars-Rev', 'contact-high-school','contact-primary-school']

    DATASET = DATASET_LIST[0]  # Choose your dataset to run
    METHOD = "hyperdegree"   # options: 'degree', 'hyperdegree', add your own method here
    K = 10             # number of seeds to select

    graph, communities = load_graph(DATASET, None, verbose=True)

    seeds = select_seeds(graph, METHOD, K)
    print(f"\nSelected seeds by {METHOD} (K={K}): {seeds}")

    print(f"\nCommunity loaded: {len(communities)} communities")



if __name__ == "__main__":
    main()
