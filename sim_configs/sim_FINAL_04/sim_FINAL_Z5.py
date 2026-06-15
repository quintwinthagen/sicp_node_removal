from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Bars-Rev'],
    p_values=[0.4],
    beta=0.02,
    tau=None,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    EMPTY_STRATEGY,
    Strategy("RAND", random_based_removal, repeat=100),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)