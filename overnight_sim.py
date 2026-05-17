# Comprehensive multi-dataset, multi-p comparison with file saves
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from main import load_graph_and_communities, run_configured_sicp_intermediates
from removal import (
    degree_based_removal,
    hyperdegree_based_removal,
    random_based_removal,
    ept_bridge_out_strength_removal,
    ept_bridge_in_strength_removal,
    ept_boundary_strength_removal,
    ept_bridge_out_fraction_removal,
    ept_bridge_broker_removal,
    ept_boundary_ratio_removal,
    ept_broker_strength,
    ept_pagerank_removal,
    ept_total_strength,
    ept_out_strength,
    remove_nodes,
)


def run_sims():
    # Parameters
    DATASETS = ['Algebra', 'Bars-Rev', 'Music-Rev', 'Restaurants-Rev']
    P_VALUES = [0.05, 0.10, 0.20]
    BETA = 0.02
    T = 25
    RUNS = 10
    rng_seed = 175
    seed_iterations = None  # unbounded

    # Output directory
    out_dir = Path("outputs/comprehensive_sims")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strategy wrapper utilities

    def _wrap_no_comm(func):
        return lambda graph, communities, K: func(graph, K)


    def _wrap_beta(func):
        return lambda graph, communities, K: func(graph, BETA, K)


    def _wrap_comm_beta(func):
        return lambda graph, communities, K: func(graph, communities, BETA, K)


    # Strategy list with wrappers so all removal functions can be called uniformly
    STRATEGIES = [
        ("degree", _wrap_no_comm(degree_based_removal)),
        ("hyperdegree", _wrap_no_comm(hyperdegree_based_removal)),
        ("random", _wrap_no_comm(random_based_removal)),
        ("ept_out_strength", _wrap_beta(ept_out_strength)),
        ("ept_total_strength", _wrap_beta(ept_total_strength)),
        ("ept_pagerank", _wrap_beta(ept_pagerank_removal)),
        ("ept_broker_strength", _wrap_beta(ept_broker_strength)),
        ("ept_bridge_out_strength", _wrap_comm_beta(ept_bridge_out_strength_removal)),
        ("ept_bridge_in_strength", _wrap_comm_beta(ept_bridge_in_strength_removal)),
        ("ept_boundary_strength", _wrap_comm_beta(ept_boundary_strength_removal)),
        ("ept_bridge_out_fraction", _wrap_comm_beta(ept_bridge_out_fraction_removal)),
        ("ept_bridge_broker", _wrap_comm_beta(ept_bridge_broker_removal)),
        ("ept_boundary_ratio", _wrap_comm_beta(ept_boundary_ratio_removal)),
    ]

    # CSV header
    csv_header = ['dataset', 'p', 'strategy', 'K', 'remaining_nodes', 'timestep', 'mean_infected']

    overall_start = time.time()
    total_tasks = len(DATASETS) * len(P_VALUES) * len(STRATEGIES)
    completed_tasks = 0

    # Process each dataset
    for dataset_idx, dataset in enumerate(DATASETS, 1):
        dataset_start = time.time()
        print(f"\n{'='*70}")
        print(f"[{dataset_idx}/{len(DATASETS)}] Dataset: {dataset}")
        print(f"{'='*70}")
        
        # Load graph once per dataset
        print(f"  Loading graph...")
        temp_graph, communities = load_graph_and_communities(dataset, tau=2, verbose=False)
        all_nodes = set.union(*temp_graph.values())
        N = len(all_nodes)
        print(f"  ✓ Loaded: {len(temp_graph)} hyperedges, {N} nodes")
        
        # Per-dataset output file
        dataset_out_file = out_dir / f"{dataset}_comprehensive_results.csv"
        
        # Write CSV header if file doesn't exist
        if not dataset_out_file.exists():
            with open(dataset_out_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
        
        # Process each p value
        for p_idx, p in enumerate(P_VALUES, 1):
            p_start = time.time()
            K = int(p * float(N))
            print(f"\n  [{p_idx}/{len(P_VALUES)}] p={p:.2f} (K={K} nodes to remove)")
            
            # Apply removal strategies
            print(f"    Computing removal strategies...")
            removed_graphs = {}
            strategy_start = time.time()
            for strat_name, strat_func in STRATEGIES:
                try:
                    nodes_to_remove = strat_func(temp_graph, communities, K)
                    removed_graphs[strat_name] = remove_nodes(temp_graph, nodes_to_remove)
                    print(f"      ✓ {strat_name}")
                except Exception as e:
                    print(f"      ✗ {strat_name}: {type(e).__name__}")
                    removed_graphs[strat_name] = None
            
            strategy_elapsed = time.time() - strategy_start
            print(f"    Removal strategies done in {strategy_elapsed:.1f}s")
            
            # Run SICP for each strategy and save results
            print(f"    Running SICP simulations...")
            rows = []
            sicp_start = time.time()
            for strat_idx, (strat_name, strat_func) in enumerate(STRATEGIES, 1):
                completed_tasks += 1
                if removed_graphs[strat_name] is None:
                    print(f"      [{strat_idx}/{len(STRATEGIES)}] {strat_name:<20} skipped (removal failed)")
                    continue
                
                g_removed = removed_graphs[strat_name]
                remaining_nodes = len(set.union(*g_removed.values())) if g_removed else 0
                
                # Run SICP with intermediate results
                try:
                    mean_counts = run_configured_sicp_intermediates(
                        g_removed,
                        seed_iterations=seed_iterations,
                        beta=BETA,
                        T=T,
                        runs=RUNS,
                        rng_seed=rng_seed,
                    )
                    
                    # Ensure we have exactly T+1 timesteps
                    if len(mean_counts) < T + 1:
                        mean_counts = mean_counts + [mean_counts[-1]] * (T + 1 - len(mean_counts))
                    elif len(mean_counts) > T + 1:
                        mean_counts = mean_counts[:T+1]
                    
                    # Save all timesteps to CSV
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
            
            sicp_elapsed = time.time() - sicp_start
            
            # Append rows to CSV
            if rows:
                with open(dataset_out_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writerows(rows)
                print(f"    ✓ Saved {len(rows)} results in {sicp_elapsed:.1f}s")
            
            p_elapsed = time.time() - p_start
            print(f"    Total for p={p:.2f}: {p_elapsed:.1f}s")
        
        dataset_elapsed = time.time() - dataset_start
        print(f"  Dataset {dataset} complete in {dataset_elapsed:.1f}s")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"✓ All simulations complete!")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}m)")
    print(f"  Results saved to: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_sims()