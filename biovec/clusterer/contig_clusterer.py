"""
Contig-level clustering of embedding dictionaries.

Accepts a dictionary mapping keys (e.g. contig IDs) to 2-D embedding matrices,
computes an all-pairs similarity matrix, builds a KNN graph, and runs Leiden
community detection.  Produces cluster labels, per-cluster quality statistics,
and optional TSV output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pairwise_cosine_similarity(
    A: np.ndarray,
    B: np.ndarray,
    eps: float = 1e-12,
    return_matrix: bool = False,
):
    """
    Cosine similarity between rows of *A* and rows of *B*.

    Returns
    -------
    sim : (n, m) array – only when *return_matrix* is True
    max_mean : float – mean of per-row maxima
    argmax_idx : (n,) array – best-match index in *B* for each row of *A*
    one2one : float – fraction of unique best-match indices
    max_vals : (n,) array – per-row maximum similarities
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D matrices")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)

    sim = A_norm @ B_norm.T
    argmax_idx = sim.argmax(axis=1)
    max_vals = sim[np.arange(sim.shape[0]), argmax_idx]
    max_mean = float(max_vals.mean()) if max_vals.size else float("nan")
    one2one = len(set(argmax_idx.tolist())) / max(len(argmax_idx), 1)

    if return_matrix:
        return sim, max_mean, argmax_idx, one2one, max_vals
    return max_mean, argmax_idx, one2one, max_vals


def all_pairs_similarity(
    embs_dict: Dict[str, np.ndarray],
    keys: Optional[List[str]] = None,
    eps: float = 1e-12,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Compute all-pairs max-mean cosine similarity and one-to-one ratio.

    For each ordered pair *(i, j)* the *max_mean* entry is the average over
    rows of *A_i* of the maximum cosine similarity to any row of *B_j*.

    Returns
    -------
    keys : list[str]
    max_mean_mat : (N, N) float32 array
    one2one_mat  : (N, N) float32 array
    """
    if keys is None:
        keys = list(embs_dict.keys())
    else:
        keys = list(keys)

    N = len(keys)
    max_mean_mat = np.empty((N, N), dtype=np.float32)
    one2one_mat = np.empty((N, N), dtype=np.float32)

    normed = {}
    nrows = {}
    for k in keys:
        X = np.asarray(embs_dict[k], dtype=np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        normed[k] = X
        nrows[k] = X.shape[0]

    for i, ki in enumerate(keys):
        A = normed[ki]
        nA = nrows[ki]
        if nA == 0:
            max_mean_mat[i, :] = np.nan
            one2one_mat[i, :] = np.nan
            continue
        for j, kj in enumerate(keys):
            B = normed[kj]
            if B.shape[0] == 0:
                max_mean_mat[i, j] = np.nan
                one2one_mat[i, j] = np.nan
                continue
            sim = A @ B.T
            argmax_idx = sim.argmax(axis=1)
            max_vals = sim[np.arange(nA), argmax_idx]
            max_mean_mat[i, j] = max_vals.mean()
            one2one_mat[i, j] = len(np.unique(argmax_idx)) / nA

    return keys, max_mean_mat, one2one_mat


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ContigClusterResult:
    """Output of :class:`ContigClusterer`."""
    keys: List[str]
    labels: np.ndarray
    n_clusters: int
    modularity: float
    sim_matrix: np.ndarray
    cluster_stats: List[Dict]
    summary: Dict


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContigClusterer:
    """
    Cluster items in an embeddings dictionary using Leiden community detection.

    Parameters
    ----------
    k : int
        Number of nearest neighbours per node in the KNN graph.
    sim_threshold : float
        Minimum similarity to retain an edge (edges below this are pruned).
    resolution : float
        Leiden resolution parameter (higher → more / smaller clusters).
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        k: int = 5,
        sim_threshold: float = 0.85,
        resolution: float = 10.0,
        verbose: bool = True,
    ):
        self.k = k
        self.sim_threshold = sim_threshold
        self.resolution = resolution
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        embs_dict: Dict[str, np.ndarray],
    ) -> ContigClusterResult:
        """
        Cluster the items in *embs_dict*.

        Parameters
        ----------
        embs_dict : dict[str, ndarray]
            Mapping of item key → 2-D embedding matrix (n_proteins, dim).

        Returns
        -------
        ContigClusterResult
        """
        keys, sim_matrix, _ = all_pairs_similarity(embs_dict)
        n = len(keys)

        if self.verbose:
            print(f"[ContigClusterer] {n} items, sim_matrix {sim_matrix.shape}")

        # Build KNN graph from the similarity matrix
        labels, modularity = self._leiden_cluster(sim_matrix, n)

        n_clusters = int(labels.max() + 1) if len(labels) else 0
        if self.verbose:
            print(f"[ContigClusterer] {n_clusters} clusters (modularity={modularity:.4f})")

        cluster_stats = self._compute_cluster_stats(keys, labels, sim_matrix)
        summary = self._compute_summary(cluster_stats)

        return ContigClusterResult(
            keys=keys,
            labels=labels,
            n_clusters=n_clusters,
            modularity=modularity,
            sim_matrix=sim_matrix,
            cluster_stats=cluster_stats,
            summary=summary,
        )

    def cluster_and_save(
        self,
        embs_dict: Dict[str, np.ndarray],
        output_prefix: str,
    ) -> ContigClusterResult:
        """
        Cluster then write three TSV files:

        * ``{output_prefix}.clusters.tsv`` – per-key cluster assignment
        * ``{output_prefix}.cluster_stats.tsv`` – per-cluster quality
        * ``{output_prefix}.summary.tsv`` – aggregate summary
        """
        result = self.cluster(embs_dict)
        self._write_clusters_tsv(result, f"{output_prefix}.clusters.tsv")
        self._write_stats_tsv(result.cluster_stats, f"{output_prefix}.cluster_stats.tsv")
        self._write_summary_tsv(result.summary, f"{output_prefix}.summary.tsv")
        if self.verbose:
            print(f"[ContigClusterer] Outputs → {output_prefix}.*")
        return result

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _leiden_cluster(
        self,
        sim_matrix: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, float]:
        S = sim_matrix.copy()
        np.fill_diagonal(S, 0.0)
        k = min(self.k, n - 1) if n > 1 else 0

        if k == 0:
            return np.zeros(n, dtype=int), 0.0

        rows, cols, vals = [], [], []
        for i in range(n):
            idx = np.argpartition(S[i], -k)[-k:]
            rows.extend([i] * k)
            cols.extend(idx.tolist())
            vals.extend(S[i, idx].tolist())

        vals_arr = np.asarray(vals)
        mask = vals_arr > self.sim_threshold
        sparse = sp.coo_matrix(
            (vals_arr[mask], (np.asarray(rows)[mask], np.asarray(cols)[mask])),
            shape=(n, n),
        )
        sparse = sparse.tocsr()
        sym = sparse.maximum(sparse.T)
        coo = sym.tocoo()

        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        weights = coo.data.tolist()

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution,
        )
        return np.array(partition.membership), partition.modularity

    # ------------------------------------------------------------------
    # Quality statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cluster_stats(
        keys: List[str],
        labels: np.ndarray,
        sim_matrix: np.ndarray,
    ) -> List[Dict]:
        """Per-cluster min / max / mean / std of intra-cluster similarity."""
        stats: List[Dict] = []
        for cid in range(int(labels.max()) + 1 if len(labels) else 0):
            idx = np.where(labels == cid)[0]
            member_keys = [keys[i] for i in idx]
            size = len(idx)

            if size <= 1:
                stats.append({
                    "cluster_id": cid,
                    "size": size,
                    "mean_similarity": 1.0 if size == 1 else float("nan"),
                    "min_similarity": 1.0 if size == 1 else float("nan"),
                    "max_similarity": 1.0 if size == 1 else float("nan"),
                    "std_similarity": 0.0 if size == 1 else float("nan"),
                    "members": member_keys,
                })
                continue

            sub = sim_matrix[np.ix_(idx, idx)]
            triu = sub[np.triu_indices(size, k=1)]
            stats.append({
                "cluster_id": cid,
                "size": size,
                "mean_similarity": float(triu.mean()),
                "min_similarity": float(triu.min()),
                "max_similarity": float(triu.max()),
                "std_similarity": float(triu.std()),
                "members": member_keys,
            })
        return stats

    @staticmethod
    def _compute_summary(cluster_stats: List[Dict]) -> Dict:
        """Aggregate quality across all clusters."""
        if not cluster_stats:
            return {}

        sizes = np.array([c["size"] for c in cluster_stats], dtype=np.float32)
        means = np.array([c["mean_similarity"] for c in cluster_stats
                          if not np.isnan(c["mean_similarity"])])
        mins = np.array([c["min_similarity"] for c in cluster_stats
                         if not np.isnan(c["min_similarity"])])

        summary: Dict = {
            "n_clusters": len(cluster_stats),
            "total_items": int(sizes.sum()),
            "mean_cluster_size": float(sizes.mean()),
            "median_cluster_size": float(np.median(sizes)),
        }

        if len(means):
            summary.update({
                "avg_mean_similarity": float(means.mean()),
                "min_mean_similarity": float(means.min()),
                "max_mean_similarity": float(means.max()),
                "weighted_mean_similarity": float(
                    np.average(means, weights=sizes[:len(means)])
                ),
            })
        if len(mins):
            summary.update({
                "avg_min_similarity": float(mins.mean()),
                "min_min_similarity": float(mins.min()),
                "max_min_similarity": float(mins.max()),
            })

        return summary

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @staticmethod
    def _write_clusters_tsv(result: ContigClusterResult, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("key\tcluster_id\n")
            for key, label in zip(result.keys, result.labels):
                f.write(f"{key}\t{label}\n")

    @staticmethod
    def _write_stats_tsv(cluster_stats: List[Dict], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = ["cluster_id", "size", "mean_similarity", "min_similarity",
                "max_similarity", "std_similarity", "members"]
        with open(p, "w") as f:
            f.write("\t".join(cols) + "\n")
            for row in cluster_stats:
                members_str = ",".join(row.get("members", []))
                vals = [str(row.get(c, "")) if c != "members" else members_str
                        for c in cols]
                f.write("\t".join(vals) + "\n")

    @staticmethod
    def _write_summary_tsv(summary: Dict, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("metric\tvalue\n")
            for k, v in summary.items():
                f.write(f"{k}\t{v}\n")


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def cluster_contigs(
    input_dir: str,
    output_prefix: str,
    *,
    h5_glob: str = "*.h5",
    matrix_key: str = "matrix",
    k: int = 5,
    sim_threshold: float = 0.85,
    resolution: float = 10.0,
    verbose: bool = True,
) -> ContigClusterResult:
    """
    Load H5 embedding files from a directory and cluster them with Leiden.

    Each H5 file is treated as one item (e.g. one contig).  The file stem
    is used as the dictionary key.

    Parameters
    ----------
    input_dir : str
        Directory containing H5 embedding files.
    output_prefix : str
        Path prefix for output files (``*.clusters.tsv``,
        ``*.cluster_stats.tsv``, ``*.summary.tsv``).
    h5_glob : str
        Glob pattern for H5 files inside *input_dir* (default ``"*.h5"``).
    matrix_key : str
        Dataset name inside each H5 file that holds the embedding matrix
        (default ``"matrix"``).
    k : int
        KNN neighbours per node.
    sim_threshold : float
        Minimum similarity to keep an edge.
    resolution : float
        Leiden resolution parameter.
    verbose : bool
        Print progress.

    Returns
    -------
    ContigClusterResult
    """
    import h5py

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    h5_files = sorted(input_path.glob(h5_glob))
    if not h5_files:
        raise FileNotFoundError(
            f"No files matching '{h5_glob}' in {input_dir}"
        )

    if verbose:
        print(f"[cluster_contigs] Loading {len(h5_files)} H5 files from {input_dir}")

    embs_dict: Dict[str, np.ndarray] = {}
    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            if matrix_key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Dataset '{matrix_key}' not found in {fp}. "
                    f"Available: {available}"
                )
            embs_dict[fp.stem] = f[matrix_key][:]

    if verbose:
        total_rows = sum(v.shape[0] for v in embs_dict.values())
        dim = next(iter(embs_dict.values())).shape[1]
        print(f"[cluster_contigs] {len(embs_dict)} contigs, "
              f"{total_rows} total embeddings, dim={dim}")

    clusterer = ContigClusterer(
        k=k,
        sim_threshold=sim_threshold,
        resolution=resolution,
        verbose=verbose,
    )
    return clusterer.cluster_and_save(embs_dict, output_prefix)
