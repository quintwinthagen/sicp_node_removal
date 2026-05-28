from core.removal import (
    make_adaptive_removal,

    # Baselines / non-community
    degree_based_removal,
    hyperdegree_based_removal,
    random_based_removal,
    avg_hyperedge_size_removal,

    # Community-based (no beta)
    count_communities_in_hedges_removal,
    count_ic_hedges,
    responsibility_weighted_hdeg_removal,
    filtered_hyperdegree_removal,

    # EPT-based (beta, some with communities)
    ept_bridge_out_strength_removal,
    ept_bridge_in_strength_removal,
    ept_bridge_in_and_out_total,
    ept_boundary_strength_removal,
    ept_bridge_out_fraction_removal,
    ept_bridge_broker_removal,
    ept_boundary_ratio_removal,

    # EPT-based (beta, no communities)
    ept_broker_strength,
    ept_pagerank_removal,
    ept_total_strength,
    ept_out_strength,
)
from scripts.sim_runner import *

# Music Rev
# Bars Rev
# intercommunity hyperedge count

parameters = SimParameters(
    datasets=['Algebra', 'Music-Rev'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    # -------------------------
    # Baselines / no communities / no beta
    # -------------------------
    Strategy("degree", degree_based_removal, need_beta=False, need_community=False),
    Strategy("degree_greedy", make_adaptive_removal(degree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("hyperdegree", hyperdegree_based_removal, need_beta=False, need_community=False),
    Strategy("hyperdegree_greedy", make_adaptive_removal(hyperdegree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("random", random_based_removal, need_beta=False, need_community=False),
    Strategy("random_greedy", make_adaptive_removal(random_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("avg_hyperedge_size", avg_hyperedge_size_removal, need_beta=False, need_community=False),
    Strategy("avg_hyperedge_size_greedy", make_adaptive_removal(avg_hyperedge_size_removal, batch=1), need_beta=False, need_community=False),

    # -------------------------
    # Community-based / no beta
    # -------------------------
    Strategy("count_communities_in_hedges", count_communities_in_hedges_removal, need_beta=False, need_community=True),
    Strategy("count_communities_in_hedges_greedy", make_adaptive_removal(count_communities_in_hedges_removal, batch=1), need_beta=False, need_community=True),

    Strategy("count_ic_hedges", count_ic_hedges, need_beta=False, need_community=True),
    Strategy("count_ic_hedges_greedy", make_adaptive_removal(count_ic_hedges, batch=1), need_beta=False, need_community=True),

    Strategy("responsibility_weighted_hdeg", responsibility_weighted_hdeg_removal, need_beta=False, need_community=True),
    Strategy("responsibility_weighted_hdeg_greedy", make_adaptive_removal(responsibility_weighted_hdeg_removal, batch=1), need_beta=False, need_community=True),

    Strategy("filtered_hyperdegree", filtered_hyperdegree_removal, need_beta=False, need_community=True),
    Strategy("filtered_hyperdegree_greedy", make_adaptive_removal(filtered_hyperdegree_removal, batch=1), need_beta=False, need_community=True),

    # -------------------------
    # EPT-based / beta + communities
    # -------------------------
    Strategy("ept_bridge_out_strength", ept_bridge_out_strength_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_out_strength_greedy", make_adaptive_removal(ept_bridge_out_strength_removal, batch=1), need_beta=True, need_community=True),

    Strategy("ept_bridge_in_strength", ept_bridge_in_strength_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_in_strength_greedy", make_adaptive_removal(ept_bridge_in_strength_removal, batch=1), need_beta=True, need_community=True),

    Strategy("ept_bridge_in_out_total", ept_bridge_in_and_out_total, need_beta=True, need_community=True),
    Strategy("ept_bridge_in_out_total_greedy", make_adaptive_removal(ept_bridge_in_and_out_total, batch=1), need_beta=True, need_community=True),

    Strategy("ept_boundary_strength", ept_boundary_strength_removal, need_beta=True, need_community=True),
    Strategy("ept_boundary_strength_greedy", make_adaptive_removal(ept_boundary_strength_removal, batch=1), need_beta=True, need_community=True),

    Strategy("ept_bridge_out_fraction", ept_bridge_out_fraction_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_out_fraction_greedy", make_adaptive_removal(ept_bridge_out_fraction_removal, batch=1), need_beta=True, need_community=True),

    Strategy("ept_bridge_broker", ept_bridge_broker_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_broker_greedy", make_adaptive_removal(ept_bridge_broker_removal, batch=1), need_beta=True, need_community=True),

    Strategy("ept_boundary_ratio", ept_boundary_ratio_removal, need_beta=True, need_community=True),
    Strategy("ept_boundary_ratio_greedy", make_adaptive_removal(ept_boundary_ratio_removal, batch=1), need_beta=True, need_community=True),

    # -------------------------
    # EPT-based / beta only (no communities)
    # -------------------------
    Strategy("ept_broker_strength", ept_broker_strength, need_beta=True, need_community=False),
    Strategy("ept_broker_strength_greedy", make_adaptive_removal(ept_broker_strength, batch=1), need_beta=True, need_community=False),

    Strategy("ept_pagerank", ept_pagerank_removal, need_beta=True, need_community=False),
    Strategy("ept_pagerank_greedy", make_adaptive_removal(ept_pagerank_removal, batch=1), need_beta=True, need_community=False),

    Strategy("ept_total_strength", ept_total_strength, need_beta=True, need_community=False),
    Strategy("ept_total_strength_greedy", make_adaptive_removal(ept_total_strength, batch=1), need_beta=True, need_community=False),

    Strategy("ept_out_strength", ept_out_strength, need_beta=True, need_community=False),
    Strategy("ept_out_strength_greedy", make_adaptive_removal(ept_out_strength, batch=1), need_beta=True, need_community=False),
]


if __name__ == "__main__":
    run_sims(parameters, strategies)