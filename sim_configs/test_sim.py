from ..removal import ept_total_strength
from ..sim_runner import *

parameters = SimParams(
    datasets=['Geometry'],
    p_values=[0.2],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("ept_total_strength", ept_total_strength, need_beta=True, need_community=False)
]

if __name__ == "__main__":
    run_sims(parameters, strategies)