"""
Statistics and metrics functions for cluster analysis.

Provides functions to compute cohesion statistics, sequence identity,
and cluster representatives.
"""

from typing import Dict, List, Optional

import numpy as np

from ..seqaligner.pairwise_aligner import pairwise_identity_matrix


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity computation.
    
    Args:
        embeddings: Array of shape (N, D).
        
    Returns:
        L2-normalized embeddings of same shape.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def compute_cluster_cohesion(
    cluster_embeddings: np.ndarray,
    cluster_ids: Optional[List[str]] = None,
    sequences: Optional[Dict[str, str]] = None,
    cluster_id: int = 0,
    max_samples: int = 1000,
    sequence_alignment: bool = False,
) -> Dict:
    """
    Compute cohesion statistics for a single cluster.
    
    Args:
        cluster_embeddings: Embeddings for cluster members, shape (N, D).
        cluster_ids: List of protein IDs for cluster members (for sequence lookup).
        sequences: Dict mapping protein ID -> sequence (for alignment stats).
        cluster_id: Cluster ID for output dict.
        max_samples: Maximum samples for pairwise computation.
        sequence_alignment: If True, also compute pairwise sequence identity stats.
                    
    Returns:
        Dict with: size, mean/min/max/std_similarity, radius, mean_centroid_similarity,
                   and optionally sequence identity stats if sequence_alignment=True.
    """
    orig_size = len(cluster_embeddings)
    
    # Handle edge cases
    if orig_size == 0:
        base_stats = {
            "cluster_id": cluster_id, "size": 0,
            "mean_similarity": np.nan, "min_similarity": np.nan,
            "max_similarity": np.nan, "std_similarity": np.nan,
            "radius": np.nan, "mean_centroid_similarity": np.nan,
        }
        if sequence_alignment:
            base_stats.update({
                "mean_seq_identity": np.nan, "min_seq_identity": np.nan,
                "max_seq_identity": np.nan, "std_seq_identity": np.nan,
            })
        return base_stats
    
    if orig_size == 1:
        base_stats = {
            "cluster_id": cluster_id, "size": 1,
            "mean_similarity": 1.0, "min_similarity": 1.0,
            "max_similarity": 1.0, "std_similarity": 0.0,
            "radius": 0.0, "mean_centroid_similarity": 1.0,
        }
        if sequence_alignment:
            base_stats.update({
                "mean_seq_identity": 100.0, "min_seq_identity": 100.0,
                "max_seq_identity": 100.0, "std_seq_identity": 0.0,
            })
        return base_stats
    
    # Subsample if needed
    embs = cluster_embeddings
    sample_ids = cluster_ids
    if orig_size > max_samples:
        sample_idx = np.random.choice(orig_size, max_samples, replace=False)
        embs = cluster_embeddings[sample_idx]
        if cluster_ids is not None:
            sample_ids = [cluster_ids[i] for i in sample_idx]
    n = len(embs)
    
    # Normalize for cosine similarity
    X = normalize_embeddings(embs)
    
    # Pairwise cosine similarities
    sim_matrix = X @ X.T
    pairwise_sims = sim_matrix[np.triu_indices(n, k=1)]
    
    # Centroid similarities
    centroid = normalize_embeddings(X.mean(axis=0, keepdims=True))
    centroid_sims = (X @ centroid.T).flatten()
    
    result_dict = {
        "cluster_id": cluster_id,
        "size": orig_size,
        "mean_similarity": float(pairwise_sims.mean()),
        "min_similarity": float(pairwise_sims.min()),
        "max_similarity": float(pairwise_sims.max()),
        "std_similarity": float(pairwise_sims.std()) if n > 2 else 0.0,
        "radius": float(1.0 - centroid_sims.min()),
        "mean_centroid_similarity": float(centroid_sims.mean()),
    }
    
    # Compute sequence alignment stats if requested
    if sequence_alignment:
        result_dict.update(_compute_sequence_identity_stats(sample_ids, sequences, cluster_id))
    
    return result_dict


def _compute_sequence_identity_stats(
    cluster_ids: Optional[List[str]],
    sequences: Optional[Dict[str, str]],
    cluster_id: int = 0,
) -> Dict:
    """
    Compute pairwise sequence identity statistics.
    
    Args:
        cluster_ids: List of protein IDs in the cluster.
        sequences: Dict mapping protein ID -> sequence.
        cluster_id: Cluster ID for warning messages.
        
    Returns:
        Dict with mean/min/max/std_seq_identity.
    """
    nan_result = {
        "mean_seq_identity": np.nan, "min_seq_identity": np.nan,
        "max_seq_identity": np.nan, "std_seq_identity": np.nan,
    }
    
    if sequences is None or cluster_ids is None:
        return nan_result
    
    # Get sequences for cluster members
    seq_list = []
    for pid in cluster_ids:
        if pid in sequences:
            seq_list.append(sequences[pid])
    
    if len(seq_list) < 2:
        return nan_result
    
    try:
        id_matrix = pairwise_identity_matrix(seq_list)
        pairwise_ids = id_matrix[np.triu_indices(len(seq_list), k=1)]
        return {
            "mean_seq_identity": float(pairwise_ids.mean()),
            "min_seq_identity": float(pairwise_ids.min()),
            "max_seq_identity": float(pairwise_ids.max()),
            "std_seq_identity": float(pairwise_ids.std()) if len(seq_list) > 2 else 0.0,
        }
    except Exception:
        return nan_result


def compute_cohesion_summary(cohesion_stats: List[Dict]) -> Dict:
    """
    Summarize cohesion statistics across all clusters.
    
    Args:
        cohesion_stats: List of cohesion dicts from compute_cluster_cohesion().
        
    Returns:
        Dict with aggregate statistics across clusters.
    """
    if not cohesion_stats:
        return {}
    
    sizes = np.array([c["size"] for c in cohesion_stats], dtype=np.float32)
    mean_sims = np.array([c["mean_similarity"] for c in cohesion_stats if not np.isnan(c["mean_similarity"])])
    min_sims = np.array([c["min_similarity"] for c in cohesion_stats if not np.isnan(c["min_similarity"])])
    radii = np.array([c["radius"] for c in cohesion_stats if not np.isnan(c["radius"])])
    
    summary = {
        "n_clusters": len(cohesion_stats),
        "total_samples": int(sizes.sum()),
        "mean_cluster_size": float(sizes.mean()),
        "median_cluster_size": float(np.median(sizes)),
        "weighted_mean_similarity": float(np.average(mean_sims, weights=sizes[:len(mean_sims)])),
        # Mean similarity stats
        "avg_mean_similarity": float(mean_sims.mean()),
        "min_mean_similarity": float(mean_sims.min()) if len(mean_sims) > 0 else np.nan,
        "max_mean_similarity": float(mean_sims.max()) if len(mean_sims) > 0 else np.nan,
        # Min similarity stats (cluster diameter)
        "avg_min_similarity": float(min_sims.mean()),
        "min_min_similarity": float(min_sims.min()) if len(min_sims) > 0 else np.nan,
        "max_min_similarity": float(min_sims.max()) if len(min_sims) > 0 else np.nan,
        # Radius stats
        "avg_radius": float(radii.mean()),
        "min_radius": float(radii.min()) if len(radii) > 0 else np.nan,
        "max_radius": float(radii.max()) if len(radii) > 0 else np.nan,
    }
    
    # Add sequence identity stats if available
    if cohesion_stats and "mean_seq_identity" in cohesion_stats[0]:
        seq_ids = np.array([c["mean_seq_identity"] for c in cohesion_stats if not np.isnan(c["mean_seq_identity"])])
        min_seq_ids = np.array([c["min_seq_identity"] for c in cohesion_stats if not np.isnan(c["min_seq_identity"])])
        if len(seq_ids) > 0:
            summary.update({
                "avg_mean_seq_identity": float(seq_ids.mean()),
                "min_mean_seq_identity": float(seq_ids.min()),
                "max_mean_seq_identity": float(seq_ids.max()),
                "weighted_mean_seq_identity": float(np.average(seq_ids, weights=sizes[:len(seq_ids)])),
            })
        if len(min_seq_ids) > 0:
            summary.update({
                "avg_min_seq_identity": float(min_seq_ids.mean()),
                "min_min_seq_identity": float(min_seq_ids.min()),
                "max_min_seq_identity": float(min_seq_ids.max()),
            })
    
    return summary


def find_cluster_representative(
    cluster_embeddings: np.ndarray,
    cluster_ids: List[str],
    cluster_id: int = 0,
) -> Dict:
    """
    Find the representative member of a cluster (closest to centroid).
    
    Args:
        cluster_embeddings: Embeddings for cluster members, shape (N, D).
        cluster_ids: List of protein IDs for cluster members.
        cluster_id: Cluster ID for output dict.
        
    Returns:
        Dict with representative id, local index, and similarity to centroid.
    """
    n = len(cluster_embeddings)
    
    if n == 0:
        return {
            "cluster_id": cluster_id,
            "representative_id": None,
            "representative_local_idx": None,
            "centroid_similarity": np.nan,
        }
    
    if n == 1:
        return {
            "cluster_id": cluster_id,
            "representative_id": cluster_ids[0] if cluster_ids else None,
            "representative_local_idx": 0,
            "centroid_similarity": 1.0,
        }
    
    # Normalize embeddings
    X = normalize_embeddings(cluster_embeddings)
    
    # Compute centroid and find member closest to it
    centroid = normalize_embeddings(X.mean(axis=0, keepdims=True))
    centroid_sims = (X @ centroid.T).flatten()
    best_local_idx = int(np.argmax(centroid_sims))
    
    return {
        "cluster_id": cluster_id,
        "representative_id": cluster_ids[best_local_idx] if cluster_ids else None,
        "representative_local_idx": best_local_idx,
        "centroid_similarity": float(centroid_sims[best_local_idx]),
    }

