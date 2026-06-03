from core.removal import random_based_removal, degree_based_removal, random_based_removal
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.04,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    EMPTY_STRATEGY,
    Strategy("random", random_based_removal, repeat=3),
    Strategy("degree", degree_based_removal, need_beta=False, need_community=False),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)