from methods.baselines import degree, hyperdegree


L_DATASETS = [
    'contact-high-school',
    'contact-primary-school'
]

EL_DATASETS = {
    'Algebra': 'cat-edge-algebra-questions',
    'Geometry': 'cat-edge-geometry-questions',
    'Restaurants-Rev': 'cat-edge-madison-restaurant-reviews',
    'Music-Rev' : 'cat-edge-music-blues-reviews',
    'Bars-Rev' : 'cat-edge-vegas-bars-reviews',
}


METHODS = {
    "Degree":      lambda K, hv: degree(hv, K),
    "Hyperdegree": lambda K, hv: hyperdegree(hv, K),
}

seed_set_degree = ['102', '16', '13', '77', '8']