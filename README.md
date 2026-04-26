# SICP Demo code

A demo for running SICP diffusion and seed selection on hypergraphs

## Contents

- `main.py`: Run SICP diffusion with a fixed seed set
- `seed_selection.py`: Select seeds using degree/hyperdegree, then optionally run SICP
- `config.py`, `data_loader.py`, `methods/`: Loads the data and implements methods to use the hypergraph data
- `data/`: Hypergraph data

## Quick Start

### 1. Run SICP with custom seeds

Edit `main.py` top constants:
```python
dataset = DATASET_LIST[0]  # e.g., 'Algebra'
BETA = 0.3
T = 25
RUNS = 10
seed_set = [1, 4, 7]
rng_seed = 79
```

Then run:
```bash
python3 main.py
```

### 2. Select seeds and run SICP

Edit `seed_selection.py` top constants:
```python
DATASET = "Algebra"
METHOD = "degree"  # or "hyperdegree"
K = 5
TAU = 2
BETA = 0.01
T = 25
RUNS = 5
RNG_SEED = 12345
RUN_SICP = True
```

Then run:
```bash
python3 seed_selection.py
```

## Available Datasets

- Node-labeled: `contact-high-school`, `contact-primary-school`
- Edge-labeled: `Algebra`, `Geometry`, `Music-Rev`, `Restaurants-Rev`, `Bars-Rev`

## Output

- Per-run infection counts and statistics
- Community summaries (if loaded)
