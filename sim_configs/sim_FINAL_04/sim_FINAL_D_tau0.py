from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry', 'Music-Rev', 'Restaurants-Rev', 'Bars-Rev', 'contact-primary-school'],
    p_values=[0.4],
    beta=0.02,
    tau=0,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("aEPT_BOS", make_adaptive_removal(ept_bridge_out_strength_removal), need_beta=True, need_community=True),
    Strategy("AHS", average_hyperedge_size_removal),
    Strategy("aAHS", make_adaptive_removal(average_hyperedge_size_removal)),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)