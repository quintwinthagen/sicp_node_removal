from core.removal import count_ic_hedges
from scripts.sim_runner import *

# Music Rev
# Bars Rev
# intercommunity hyperedge count

parameters = SimParams(
    datasets=['Music-Rev', 'Bars-Rev'],
    p_values=[0.05, 0.1, 0.2],
    beta=0.02,
    timesteps=25,
    runs=10,
    rng_seed=175,
    seed_iterations=None
)

strategies = [
    Strategy("ic_hedge_count", count_ic_hedges, need_beta=False, need_community=True)
]

if __name__ == "__main__":
    run_sims(parameters, strategies)