from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry', 'Music-Rev'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    tau=None,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("RAND", random_based_removal, need_beta=False, need_community=False, repeat=100),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)