from core.removal import *
from scripts.sim_runner import *

parameters = SimParameters(
    datasets=['contact-primary-school'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    tau=None,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("EPT_ICS", ept_intra_comm_strength, need_beta=True, need_community=True),
    Strategy("aEPT_ICS", make_adaptive_removal(ept_intra_comm_strength, batch=1), need_beta=True, need_community=True),
]

if __name__ == "__main__":
    run_sims(parameters, strategies)