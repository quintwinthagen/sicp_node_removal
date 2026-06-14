from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['Algebra', 'Geometry', 'Music-Rev', 'Restaurants-Rev', 'Bars-Rev', 'contact-primary-school'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    tau=2,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("EPT_BIS", ept_bridge_in_strength_removal, need_beta=True, need_community=True),
    Strategy("aEPT_BIS", make_adaptive_removal(ept_bridge_in_strength_removal), need_beta=True, need_community=True),
    Strategy("EPT_BOS", ept_bridge_out_strength_removal, need_beta=True, need_community=True),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)