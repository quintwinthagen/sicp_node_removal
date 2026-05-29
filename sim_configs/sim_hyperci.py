from core.removal import hyperci_removal, make_adaptive_removal, degree_based_removal, random_based_removal
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Bars-Rev'],
    p_values=[0.05, 0.1, 0.2, 0.4],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    EMPTY_STRATEGY,
    Strategy("random", random_based_removal, need_beta=False, need_community=False),
    
    Strategy("degree", degree_based_removal, need_beta=False, need_community=False),
    Strategy("degree_adaptive", make_adaptive_removal(degree_based_removal, batch=1), need_beta=False, need_community=False),

    Strategy("hyperci", hyperci_removal, need_beta=False, need_community=False),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)