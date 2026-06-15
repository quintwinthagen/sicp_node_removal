from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry', 'Music-Rev', 'Restaurants-Rev', 'Bars-Rev', 'contact-primary-school'],
    p_values=[0.4],
    beta=0.02,
    tau=None,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("DEG", degree_based_removal),
    Strategy("aDEG", make_adaptive_removal(degree_based_removal)),
    Strategy("HDEG", hyperdegree_based_removal),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)