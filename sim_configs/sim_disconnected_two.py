from core.removal import hyperdegree_based_removal, make_adaptive_removal, degree_based_removal, ept_total_strength, ept_bridge_in_strength_removal, random_based_removal
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry', 'Music-Rev'],
    p_values=[0.05, 0.1, 0.2, 0.4, 0.7],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    EMPTY_STRATEGY,
    Strategy("random", random_based_removal, need_beta=False, need_community=False),
    Strategy("random_adaptive", make_adaptive_removal(random_based_removal, batch=1), need_beta=False, need_community=False),
    
    Strategy("degree", degree_based_removal, need_beta=False, need_community=False),
    Strategy("degree_adaptive", make_adaptive_removal(degree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("hyperdegree", hyperdegree_based_removal, need_beta=False, need_community=False),
    Strategy("hyperdegree_adaptive", make_adaptive_removal(hyperdegree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("ept_bridge_in", ept_bridge_in_strength_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_in_adaptive", make_adaptive_removal(ept_bridge_in_strength_removal, batch=1), need_beta=True, need_community=True),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)