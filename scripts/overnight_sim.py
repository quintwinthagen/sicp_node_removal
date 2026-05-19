# Comprehensive multi-dataset, multi-p comparison with file saves\
import sys
sys.stdout.reconfigure(line_buffering=True)

import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .main import load_graph_and_communities, run_configured_sicp_intermediates
from core.removal import (
    degree_based_removal,
    avg_hyperedge_size_removal,
    hyperdegree_based_removal,
    random_based_removal,
    ept_bridge_in_strength_removal,
    ept_total_strength,
    remove_nodes,
)


def run_sims():
    DATASETS = ['Music-Rev']
    P_VALUES = [0.05, 0.10, 0.20]
    BETA = 0.02
    T = 25
    RUNS = 10
    rng_seed = 175
    seed_iterations = 1000

    out_dir = Path("overnight_outputs/presentation_sims2")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _wrap_no_comm(func):
        return lambda graph, communities, K: func(graph, K)

    def _wrap_beta(func):
        return lambda graph, communities, K: func(graph, BETA, K)

    def _wrap_comm_beta(func):
        return lambda graph, communities, K: func(graph, communities, BETA, K)

    STRATEGIES = [
        # ("degree", _wrap_no_comm(degree_based_removal)),
        # ("hyperdegree", _wrap_no_comm(hyperdegree_based_removal)),
        # ("random", _wrap_no_comm(random_based_removal)),
        ("avg_hyperedge_size", _wrap_no_comm(avg_hyperedge_size_removal)),
        # ("ept_total_strength", _wrap_beta(ept_total_strength)),
        # ("ept_bridge_in_strength", _wrap_comm_beta(ept_bridge_in_strength_removal)),
    ]

    csv_header = ['dataset', 'p', 'strategy', 'K', 'remaining_nodes', 'timestep', 'mean_infected']

    overall_start = time.time()

    for dataset_idx, dataset in enumerate(DATASETS, 1):
        dataset_start = time.time()
        print("\n" + "="*70)
        print(f"[{dataset_idx}/{len(DATASETS)}] Dataset: {dataset}")
        print("="*70)

        print("  Loading graph...")
        temp_graph, communities = load_graph_and_communities(dataset, tau=2, verbose=False)
        all_nodes = set.union(*temp_graph.values())
        N = len(all_nodes)
        print(f"  OK Loaded: {len(temp_graph)} hyperedges, {N} nodes")

        dataset_out_file = out_dir / f"{dataset}_comprehensive_results.csv"

        if not dataset_out_file.exists():
            with open(dataset_out_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()

        for p_idx, p in enumerate(P_VALUES, 1):
            p_start = time.time()
            K = int(p * float(N))
            print(f"\n  [{p_idx}/{len(P_VALUES)}] p={p:.2f} (K={K} nodes to remove)")

            print("    Computing removal strategies...")
            removed_graphs = {}
            strategy_start = time.time()

            for strat_name, strat_func in STRATEGIES:
                try:
                    nodes_to_remove = strat_func(temp_graph, communities, K)
                    removed_graphs[strat_name] = remove_nodes(temp_graph, nodes_to_remove)
                    print(f"      OK {strat_name}")
                except Exception as e:
                    print(f"      FAIL {strat_name}: {type(e).__name__}")
                    removed_graphs[strat_name] = None

            print(f"    Removal strategies done in {time.time() - strategy_start:.1f}s")

            print("    Running SICP simulations...")
            rows = []
            sicp_start = time.time()

            for strat_idx, (strat_name, _) in enumerate(STRATEGIES, 1):
                if removed_graphs[strat_name] is None:
                    print(f"      [{strat_idx}/{len(STRATEGIES)}] {strat_name:<20} skipped")
                    continue

                g_removed = removed_graphs[strat_name]
                remaining_nodes = len(set.union(*g_removed.values())) if g_removed else 0

                try:
                    mean_counts = run_configured_sicp_intermediates(
                        g_removed,
                        seed_iterations=seed_iterations,
                        beta=BETA,
                        T=T,
                        runs=RUNS,
                        rng_seed=rng_seed,
                    )

                    if len(mean_counts) < T + 1:
                        mean_counts += [mean_counts[-1]] * (T + 1 - len(mean_counts))
                    elif len(mean_counts) > T + 1:
                        mean_counts = mean_counts[:T+1]

                    for t, mean_inf in enumerate(mean_counts):
                        rows.append({
                            'dataset': dataset,
                            'p': p,
                            'strategy': strat_name,
                            'K': K,
                            'remaining_nodes': remaining_nodes,
                            'timestep': t,
                            'mean_infected': mean_inf,
                        })

                    final_pct = 100.0 * mean_counts[-1] / remaining_nodes if remaining_nodes > 0 else 0.0
                    print(f"      [{strat_idx}/{len(STRATEGIES)}] {strat_name:<20} final={mean_counts[-1]:6.1f} ({final_pct:5.1f}%)")

                except Exception as e:
                    print(f"      [{strat_idx}/{len(STRATEGIES)}] {strat_name:<20} FAILED: {type(e).__name__}")

            if rows:
                with open(dataset_out_file, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=csv_header).writerows(rows)
                print(f"    OK Saved {len(rows)} results in {time.time() - sicp_start:.1f}s")

            print(f"    Total for p={p:.2f}: {time.time() - p_start:.1f}s")

        print(f"  Dataset {dataset} complete in {time.time() - dataset_start:.1f}s")

    overall_elapsed = time.time() - overall_start
    print("\n" + "="*70)
    print("OK All simulations complete!")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}m)")
    print(f"Results saved to: {out_dir}")
    print("="*70)


if __name__ == "__main__":
    run_sims()