# Comprehensive multi-dataset, multi-p comparison with file saves\
from collections import defaultdict
import sys
sys.stdout.reconfigure(line_buffering=True)

import traceback

from dataclasses import dataclass
from typing import Callable, List

import csv
import time
from pathlib import Path

from .main import load_graph_and_communities, run_configured_sicp_intermediates
from core.removal import remove_nodes

import argparse

@dataclass
class SimParameters:
    datasets: List[str]
    p_values: List[float]
    beta: float
    tau: int
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
    repeat: int = 1

EMPTY_STRATEGY = Strategy(
    name="EMPTY_STRATEGY",
    removal_func=lambda graph, K : [],
    need_community=False,
    need_beta=False
)

def run_sims(parameters: SimParameters, strategies: List[Strategy]): 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str)
    args = parser.parse_args()
    out_dir = Path(args.o)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _get_removal_func(strategy: Strategy):
        if strategy.need_beta and strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, communities, parameters.beta, K=K)
        if strategy.need_beta and not strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, parameters.beta, K=K)
        if not strategy.need_beta and strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, communities, K=K)
        if not strategy.need_beta and not strategy.need_community:
            return lambda graph, communities, K: strategy.removal_func(graph, K=K)

    STRATEGIES = [(strat.name, _get_removal_func(strat), strat.repeat) for strat in strategies]

    csv_header = ['dataset', 'p', 'strategy', 'K', 'remaining_nodes', 'timestep', 'mean_infected']

    overall_start = time.time()

    for dataset_idx, dataset in enumerate(parameters.datasets, 1):
        dataset_start = time.time()
        print("\n" + "="*70)
        print(f"[{dataset_idx}/{len(parameters.datasets)}] Dataset: {dataset}")
        print("="*70)

        print("  Loading graph...")
        temp_graph, communities = load_graph_and_communities(dataset, tau=parameters.tau, verbose=False)
        all_nodes = temp_graph.nodes
        N = len(all_nodes)

        connected_nodes = len(set.union(*temp_graph.hyperedges.values())) if temp_graph.hyperedges else 0
        print(f"  OK Loaded: {len(temp_graph.hyperedges)} hyperedges, {N} nodes, ({N - connected_nodes} disconnected), (tau: {parameters.tau})")
        print(f"Running with( BETA: {parameters.beta}, timesteps: {parameters.timesteps}, runs: {parameters.runs}, rng_seed: {parameters.rng_seed}, seed_iterations: {parameters.seed_iterations} )")

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

            for strat_name, strat_func, repeat in STRATEGIES:
                for i in range(repeat):
                    try:
                        nodes_to_remove = strat_func(temp_graph, communities, K)
                        removed_graphs[(strat_name, i)] = remove_nodes(temp_graph, nodes_to_remove)
                        nodes_in_hyperedges_count = len(set.union(*removed_graphs[(strat_name, i)].hyperedges.values())) if removed_graphs[(strat_name, i)].hyperedges else 0
                        print(f"      OK {strat_name}, removed {len(nodes_to_remove)} nodes ({len(removed_graphs[(strat_name, i)].nodes) - nodes_in_hyperedges_count} disconnected)")
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f"      FAIL {strat_name}, repeat {repeat}: {type(e).__name__}")
                        removed_graphs[(strat_name, i)] = None

            print(f"    Removal strategies done in {time.time() - strategy_start:.1f}s")

            print("    Running SICP simulations...")
            rows = []
            sicp_start = time.time()

            for strat_idx, (strat_name, _, repeats) in enumerate(STRATEGIES, 1):
                rows_per_repeat = defaultdict(list)
                for i in range(repeats):

                    if removed_graphs[(strat_name, i)] is None:
                        print(f"      [{strat_idx}/{len(STRATEGIES)}] {strat_name:<20} skipped")
                        continue

                    g_removed = removed_graphs[(strat_name, i)]
                    remaining_nodes = len(g_removed.nodes)

                    try:
                        isolated_repeat_seed = parameters.rng_seed + (i * 10000000)
                        mean_counts = run_configured_sicp_intermediates(
                            g_removed,
                            seed_iterations=parameters.seed_iterations,
                            beta=parameters.beta,
                            T=parameters.timesteps,
                            runs=parameters.runs,
                            rng_seed=isolated_repeat_seed,
                        )

                        if len(mean_counts) < parameters.timesteps + 1:
                            mean_counts += [mean_counts[-1]] * (parameters.timesteps + 1 - len(mean_counts))
                        elif len(mean_counts) > parameters.timesteps + 1:
                            mean_counts = mean_counts[:parameters.timesteps+1]

                        for t, mean_inf in enumerate(mean_counts):
                            rows_per_repeat[i].append({
                                'dataset': dataset,
                                'p': p,
                                'strategy': strat_name,
                                'K': K,
                                'remaining_nodes': remaining_nodes,
                                'timestep': t,
                                'mean_infected': mean_inf,
                            })

                        final_pct = 100.0 * mean_counts[-1] / remaining_nodes if remaining_nodes > 0 else 0.0
                        print(f"      [{strat_idx}/{len(STRATEGIES)}] [repeat {i+1}/ {repeats}] {strat_name:<20} final={mean_counts[-1]:6.1f} ({final_pct:5.1f}%)")

                    except Exception as e:
                        print(f"      [{strat_idx}/{len(STRATEGIES)}] [repeat {i+1}/{repeats}] {strat_name:<20} FAILED: {type(e).__name__}")

                if rows_per_repeat:
                    timestep_mean_infected = defaultdict(list)
                    timestep_remaining_nodes = defaultdict(list)

                    for repeat_rows in rows_per_repeat.values():
                        for row in repeat_rows:
                            t = row['timestep']
                            timestep_mean_infected[t].append(row['mean_infected'])
                            timestep_remaining_nodes[t].append(row['remaining_nodes'])

                    for t in sorted(timestep_mean_infected.keys()):
                        avg_mean_infected = sum(timestep_mean_infected[t]) / len(timestep_mean_infected[t])
                        avg_remaining_nodes = sum(timestep_remaining_nodes[t]) / len(timestep_remaining_nodes[t])
                        rows.append({
                            'dataset': dataset,
                            'p': p,
                            'strategy': strat_name,
                            'K': K,
                            'remaining_nodes': avg_remaining_nodes,
                            'timestep': t,
                            'mean_infected': avg_mean_infected,
                        })

                    if repeats > 1:
                        final_avg_infected = sum(timestep_mean_infected[parameters.timesteps]) / len(timestep_mean_infected[parameters.timesteps])
                        final_avg_pct = 100.0 * final_avg_infected / avg_remaining_nodes if avg_remaining_nodes > 0 else 0.0
                        print(f"      [avg over {repeats} repeats] {strat_name:<20} final={final_avg_infected:6.1f} ({final_avg_pct:5.1f}%)")

                        
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