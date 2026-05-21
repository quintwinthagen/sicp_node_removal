# Comprehensive multi-dataset, multi-p comparison with file saves\
import sys
sys.stdout.reconfigure(line_buffering=True)

from dataclasses import dataclass
from typing import Callable, List, NamedTuple

import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .main import load_graph_and_communities, run_configured_sicp_intermediates
from core.removal import remove_nodes

import argparse

@dataclass
class SimParameters:
    datasets: List[str]
    p_values: List[float]
    beta: float
    timesteps: int
    runs: int
    rng_seed: int
    seed_iterations: int

@dataclass
class Strategy:
    name: str
    removal_func: Callable[..., List[str]]
    need_community: bool = False
    need_beta: bool = False


def run_sims(parameters: SimParameters, strategies: List[Strategy]):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str)
    args = parser.parse_args()
    out_dir = Path(args.o)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _get_removal_func(strategy: Strategy):
        if strategy.need_beta and strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, communities, parameters.beta, K)
        if strategy.need_beta and not strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, parameters.beta, K)
        if not strategy.need_beta and strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, communities, K)
        if not strategy.need_beta and not strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, K)

    STRATEGIES = [(strat.name, _get_removal_func(strat)) for strat in strategies]

    csv_header = ['dataset', 'p', 'strategy', 'K', 'remaining_nodes', 'timestep', 'mean_infected']

    overall_start = time.time()

    for dataset_idx, dataset in enumerate(parameters.datasets, 1):
        dataset_start = time.time()
        print("\n" + "="*70)
        print(f"[{dataset_idx}/{len(parameters.datasets)}] Dataset: {dataset}")
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

        for p_idx, p in enumerate(parameters.p_values, 1):
            p_start = time.time()
            K = int(p * float(N))
            print(f"\n  [{p_idx}/{len(parameters.p_values)}] p={p:.2f} (K={K} nodes to remove)")

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
                        seed_iterations=parameters.seed_iterations,
                        beta=parameters.beta,
                        T=parameters.timesteps,
                        runs=parameters.runs,
                        rng_seed=parameters.rng_seed,
                    )

                    if len(mean_counts) < parameters.timesteps + 1:
                        mean_counts += [mean_counts[-1]] * (parameters.timesteps + 1 - len(mean_counts))
                    elif len(mean_counts) > parameters.timesteps + 1:
                        mean_counts = mean_counts[:parameters.timesteps+1]

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