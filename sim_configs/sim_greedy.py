from core.removal import make_greedy_removal, degree_based_removal, ept_total_strength, ept_bridge_in_strength_removal
from scripts.sim_runner import *

# Music Rev
# Bars Rev
# intercommunity hyperedge count

parameters = SimParameters(
    datasets=['Algebra', 'Geometry'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("degree", degree_based_removal, need_beta=False, need_community=False),
    Strategy("degree_greedy", make_greedy_removal(degree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("ept_total_strength", ept_total_strength, need_beta=True, need_community=False),
    Strategy("ept_total_strength_greedy", make_greedy_removal(ept_total_strength, batch=1), need_beta=True, need_community=False),

    Strategy("ept_bridge_in", ept_bridge_in_strength_removal, need_beta=True, need_community=True),
    Strategy("ept_bridge_in_greedy", make_greedy_removal(ept_bridge_in_strength_removal, batch=1), need_beta=True, need_community=True),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)