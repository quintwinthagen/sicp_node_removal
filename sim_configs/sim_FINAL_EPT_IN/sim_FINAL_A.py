from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    tau=None,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("EPT_IS", ept_in_strength),
    Strategy("aEPT_IS", make_adaptive_removal(ept_in_strength)),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)