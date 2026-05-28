from collections import defaultdict
from dataclasses import dataclass, field
import os


@dataclass
class Hypergraph:
    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)


def load_hypergraph_from_txt(dataset):
    """
    Load a hypergraph from a txt file.
    Format: each line = space-separated list of node IDs in a hyperedge.
    Returns: dict of {hyperedge_id: set(nodes)}
    """
    G = {}
    with open(f"../data/unlabeled/{dataset}.txt", 'r') as f:
        for idx, line in enumerate(f):
            nodes = set(line.strip().split())
            G[idx] = nodes
    return G

def load_labeled_hypergraph(dataset, source=False):
    """
    Load a node-labeled hypergraph from three files.
    Returns:
    - hyperedges: dict[int, set[str]] — hyperedge ID to node IDs
    - clusters: dict[str, set[str]] — label name to set of node IDs
    """
    if source == False:
        pre_path = "../"
    else:
        pre_path = ""
    
    data_path = f'{pre_path}data/labeled/{dataset}'
    
    if dataset in ['contact-high-school', 'contact-primary-school']:
        # Load label names
        with open(f"{data_path}/label-names-{dataset}.txt", 'r') as f:
            label_names = [line.strip() for line in f]
        
        # Load node labels and map to label names
        node_to_label = {}
        with open(f"{data_path}/node-labels-{dataset}.txt", 'r') as f:
            for idx, line in enumerate(f):
                label_idx = int(line.strip())
                label_name = label_names[label_idx - 1]  # 1-based index
                node_to_label[str(idx)] = label_name
    
    else:
        # For senate-committees, node labels ARE the label names (no separate label-names file)
        node_to_label = {}
        with open(f"{data_path}/node-labels-{dataset}.txt", 'r') as f:
            for idx, line in enumerate(f):
                label_name = line.strip()  # Direct use of the label as name
                node_to_label[str(idx)] = label_name
    
    # Create clusters from node-to-label mapping
    clusters = defaultdict(set)
    for node, label in node_to_label.items():
        clusters[label].add(node)
    
    # Load hyperedges
    hyperedges = {}
    with open(f"{data_path}/hyperedges-{dataset}.txt", 'r') as f:
        for idx, line in enumerate(f):
            nodes = set(line.strip().split(','))
            hyperedges[idx] = nodes
    
    return hyperedges, dict(clusters)

def load_edge_labeled_hypergraph(dataset, source=False):
    """
    Method to load in hypergraph that has edge-labeled categories
    Recognisable by "cat-" in front of word
    If a node appears in a labeled hyperedge, it will be part of that community
    """
    if source==False:
        pre_path = "../"
    else:
        pre_path=""
    

    G = {}
    with open(f"{pre_path}data/labeled/{dataset}/hyperedges.txt", 'r') as f:
        for idx, line in enumerate(f):
            nodes = set(line.strip().split())
            G[idx] = nodes

    # extract hyperedge labels
    with open(f"{pre_path}data/labeled/{dataset}/hyperedge-labels.txt", 'r') as f:
        hyperedge_labels = [int(line.strip()) for line in f]
    
    # extract label identities
    with open(f"{pre_path}data/labeled/{dataset}/hyperedge-label-identities.txt", 'r') as f:
        label_names = [line.strip() for line in f]

    # Map label index to label name (1-based indexing)
    label_map = {i + 1: name for i, name in enumerate(label_names)}

    # Build clusters: community name -> set of nodes
    clusters = defaultdict(set)

    for eid, nodes in G.items():
        label_idx = hyperedge_labels[eid]
        community = label_map[label_idx]
        clusters[community].update(nodes)

    return G, dict(clusters)

# -------------------------- CD loaders --------------------------
from collections import defaultdict

def _load_cd_clusters_from_node2comm(path):
    """
    Read a file whose lines look like:  node_id : cluster_id
    Returns: clusters dict  { f"C{cid}": set(node_id_str), ... }

    - drop_unlabeled=True: skip nodes with cluster_id == 0
    """
    node2cid = {}
    with open(path, "r") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # tolerate "a: b", "a:b", "a b", "a,b"
            if ":" in line:
                left, right = line.split(":", 1)
            elif "," in line:
                left, right = line.split(",", 1)
            else:
                left, right = line.split(None, 1)  # whitespace
            node = left.strip()
            try:
                cid = int(right.strip())
            except ValueError:
                continue
            node2cid[str(node)] = cid

    # Build clusters (keys "C1","C2",...)
    clusters = defaultdict(set)
    for node, cid in node2cid.items():
        clusters[f"C{cid}"].add(node)
    return dict(clusters)


def load_labeled_hypergraph_cd(dataset, taus=(0,1,2), source=False,
                               cd_filename_tpl="cd_t{tau}_labels.txt",
                               verbose=True):
    """
    Load hyperedges and community-detection clusters for multiple tau values.

    Returns:
      hyperedges: dict[int, set[str]]
      cd_clusters: dict[int, dict[str, set[str]]]   # tau -> clusters dict (e.g., {"C1": {...}, ...})

    If you prefer multiple returns, see the example after the function.
    """
    from collections import defaultdict
    import os

    pre_path = "../" if source is False else ""
    data_path = f"{pre_path}data/labeled/{dataset}"

    # --- hyperedges (same as your original) ---
    hyperedges = {}
    with open(f"{data_path}/hyperedges-{dataset}.txt", "r") as f:
        for idx, line in enumerate(f):
            nodes = set(line.strip().split(","))
            hyperedges[idx] = nodes

    # --- community-detection clusters per tau ---
    cd_clusters = {}
    for tau in taus:
        path = os.path.join(data_path, cd_filename_tpl.format(tau=tau))
        if not os.path.exists(path):
            if verbose:
                print(f"[warn] missing CD labels for tau={tau}: {path}")
            cd_clusters[tau] = {}
            continue
        clusters_tau = _load_cd_clusters_from_node2comm(path)
        # (Optional) sanity check: warn if a community contains nodes not in hyperedges
        if verbose:
            all_nodes_h = set().union(*hyperedges.values()) if hyperedges else set()
            extra = set().union(*clusters_tau.values()) - {str(x) for x in all_nodes_h}
            if extra:
                print(f"[warn] {len(extra)} nodes in cd_t{tau} not found in hyperedges (first few): {sorted(list(extra))[:5]}")
        cd_clusters[tau] = clusters_tau

    return hyperedges, cd_clusters



def load_edge_labeled_hypergraph_cd(dataset, taus=(0,1,2), source=False,
                                    cd_filename_tpl="cd_t{tau}_labels.txt",
                                    verbose=True):
    """
    Load an edge-labeled hypergraph and, in addition, community-detection clusters
    for the given tau values.

    Returns:
      G: dict[int, set[str]]                 # hyperedge ID -> node IDs
      cd_clusters: dict[int, dict[str, set[str]]]   # tau -> clusters dict
    """
    pre_path = "../" if source is False else ""
    base = f"{pre_path}data/labeled/{dataset}"

    # --- hyperedges (exactly your original logic, just normalized to str) ---
    G = {}
    with open(f"{base}/hyperedges.txt", "r") as f:
        for idx, line in enumerate(f):
            nodes = set(str(x) for x in line.strip().split())
            G[idx] = nodes

    # --- community-detection clusters per tau (from node2comm files) ---
    # union of nodes from G for sanity checks
    nodes_in_G = set().union(*G.values()) if G else set()

    cd_clusters = {}
    for tau in taus:
        path = os.path.join(base, cd_filename_tpl.format(tau=tau))
        if not os.path.exists(path):
            if verbose:
                print(f"[warn] missing CD labels for tau={tau}: {path}")
            cd_clusters[tau] = {}
            continue

        clusters_tau = _load_cd_clusters_from_node2comm(path)

        if verbose and clusters_tau:
            nodes_in_cd = set().union(*clusters_tau.values())
            extra = nodes_in_cd - nodes_in_G
            if extra:
                sample = sorted(list(extra))[:5]
                print(f"[warn] {len(extra)} nodes in cd_t{tau} not found in hyperedges; e.g. {sample}")

        cd_clusters[tau] = clusters_tau

    return G, cd_clusters
