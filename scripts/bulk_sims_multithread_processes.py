from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import csv
import os
import random
import statistics

from main import load_graph_and_communities, run_configured_sicp_intermediates
from core.removal import (
    count_ic_hedges,
    count_communities_in_hedges_removal,
    degree_based_removal,
    filtered_hyperdegree_removal,
    hyperdegree_based_removal,
    PHG_community_influence,
    # ic_hedges_to_hdeg,
    random_based_removal,
    remove_nodes,
    responsibility_weighted_hdeg_removal,
)


def _worker(job_data):
    """Module-level worker function for ProcessPoolExecutor pickling."""
    strategy_name, p, strategy_idx, g_removed, K, dataset, beta, T, runs, rng_seed = job_data
    remaining_nodes = len(set.union(*g_removed.values()))

    mean_counts = run_configured_sicp_intermediates(
        g_removed,
        seed_iterations=None,
        beta=beta,
        T=T,
        runs=runs,
        rng_seed=rng_seed + int(100 * p) * 1000 + strategy_idx,
    )

    if len(mean_counts) < T + 1:
        tail = mean_counts[-1] if mean_counts else 0.0
        mean_counts = mean_counts + [tail] * (T + 1 - len(mean_counts))
    elif len(mean_counts) > T + 1:
        mean_counts = mean_counts[: T + 1]

    final_mean_count = float(mean_counts[-1]) if mean_counts else 0.0
    final_mean_pct = (100.0 * final_mean_count / remaining_nodes) if remaining_nodes else 0.0

    intermediate_rows = []
    for t, mean_count in enumerate(mean_counts):
        mean_pct = (100.0 * float(mean_count) / remaining_nodes) if remaining_nodes else 0.0
        intermediate_rows.append(
            {
                "dataset": dataset,
                "strategy": strategy_name,
                "p": p,
                "K_removed": K,
                "remaining_nodes": remaining_nodes,
                "timestep": t,
                "mean_infected_count": float(mean_count),
                "mean_infected_pct": mean_pct,
            }
        )

    return {
        "intermediate_rows": intermediate_rows,
        "dataset": dataset,
        "strategy": strategy_name,
        "p": p,
        "K_removed": K,
        "remaining_nodes": remaining_nodes,
        "final_mean_infected_count": final_mean_count,
        "final_mean_infected_pct": final_mean_pct,
    }


def run_many_multithreaded(
    dataset="Geometry",
    tau=2,
    p_values=(0.05, 0.10, 0.20),
    beta=0.02,
    T=25,
    runs=10,
    seed_iterations=None,
    rng_seed=175,
    max_workers=None,
    out_csv="outputs/multithreaded_final_runs.csv",
    out_intermediate_csv="outputs/multithreaded_intermediate_runs_geometry.csv",
):
    graph, communities = load_graph_and_communities(dataset, tau=tau, verbose=True)
    N = len(set.union(*graph.values()))

    # name, function, needs_communities
    strategies = [
        # ("count_communities_in_hedges", count_communities_in_hedges_removal, True),
        ("PHG_community_influence", PHG_community_influence, True),
        # ("ic_hedges_to_hdeg", ic_hedges_to_hdeg, True),
        # ("count_ic_hedges", count_ic_hedges, True),
        # ("responsibility_weighted_hdeg", responsibility_weighted_hdeg_removal, True),
        # ("filtered_hyperdegree", filtered_hyperdegree_removal, True),
        # ("hyperdegree", hyperdegree_based_removal, False),
        ("degree", degree_based_removal, False),
        ("random", random_based_removal, False),
    ]

    # Precompute removed graphs once per (strategy, p)
    removed_graphs = {}
    for p in p_values:
        K = int(p * float(N))
        for strategy_name, strategy_fn, needs_communities in strategies:
            random.seed(rng_seed + int(100 * p))
            if needs_communities:
                nodes_to_remove = strategy_fn(graph, communities, K)
            else:
                nodes_to_remove = strategy_fn(graph, K)

            g_removed = remove_nodes(graph, nodes_to_remove)
            if g_removed:
                removed_graphs[(strategy_name, p)] = (g_removed, K)

    strategy_rank = {name: idx for idx, (name, _, _) in enumerate(strategies, start=1)}

    jobs = []
    for p in p_values:
        for strategy_name, _, _ in strategies:
            if (strategy_name, p) not in removed_graphs:
                continue
            g_removed, K = removed_graphs[(strategy_name, p)]
            jobs.append((strategy_name, p, strategy_rank[strategy_name], g_removed, K, dataset, beta, T, runs, rng_seed))

    if max_workers is None:
        max_workers = max(1, min(len(jobs), os.cpu_count() or 4))

    print(f"Running {len(jobs)} jobs with {max_workers} processes")

    results = []
    intermediate_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            intermediate_results.extend(res["intermediate_rows"])
            res.pop("intermediate_rows", None)
            results.append(res)
            print(
                f"[{i}/{len(jobs)}] {res['strategy']:<30} p={res['p']:.2f} "
                f"final={res['final_mean_infected_pct']:.2f}%"
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "strategy",
        "p",
        "K_removed",
        "remaining_nodes",
        "final_mean_infected_count",
        "final_mean_infected_pct",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    inter_path = Path(out_intermediate_csv)
    inter_path.parent.mkdir(parents=True, exist_ok=True)

    inter_fields = [
        "dataset",
        "strategy",
        "p",
        "K_removed",
        "remaining_nodes",
        "timestep",
        "mean_infected_count",
        "mean_infected_pct",
    ]
    with inter_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inter_fields)
        writer.writeheader()
        writer.writerows(intermediate_results)

    print(f"Saved {len(results)} final rows to {out_path}")
    print(f"Saved {len(intermediate_results)} intermediate rows to {inter_path}")

    print("\nFinal prevalence by setting:")
    for r in sorted(results, key=lambda x: (x["p"], x["strategy"])):
        print(f"p={r['p']:.2f} | {r['strategy']:<30} final={r['final_mean_infected_pct']:.2f}%")


if __name__ == "__main__":
    run_many_multithreaded()
