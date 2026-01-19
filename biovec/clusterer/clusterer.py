"""
Clusterer module for protein embedding clustering using Leiden algorithm.

Loads a BioVecDB (FAISS index, metadata JSON, and H5 embeddings) and performs
graph-based clustering using KNN search and Leiden community detection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import igraph as ig
import leidenalg

from ..dbbuilder.biovecdb import BioVecDB, build_db
from ..dbbuilder.utils import load_sequences
from ..io.h5 import read_h5
from .stats import (
    normalize_embeddings,
    compute_cluster_cohesion as _compute_cluster_cohesion,
    compute_cohesion_summary,
    find_cluster_representative,
)


@dataclass
class ClusterResult:
    """Container for clustering results."""
    labels: np.ndarray
    n_clusters: int
    ids: np.ndarray
    descriptions: np.ndarray
    modularity: float


class Clusterer:
    """
    Cluster protein embeddings using KNN graph and Leiden algorithm.
    
    Loads a saved BioVecDB (FAISS index + metadata) and corresponding H5 file
    containing the embedding matrix. Builds a KNN graph and applies Leiden
    clustering to discover protein communities.
    
    Example:
        >>> clusterer = Clusterer("/path/to/biovecdb_output")
        >>> result = clusterer.cluster(k=30, dist_threshold=0.1, resolution=0.1)
        >>> print(f"Found {result.n_clusters} clusters")
    """
    
    def __init__(
        self,
        base_path: str,
        use_gpu: bool = True,
        metric: str = "cosine",
        verbose: bool = True,
    ):
        """
        Initialize the Clusterer with a saved BioVecDB.
        
        Args:
            base_path: Path prefix to saved BioVecDB files (without extension).
                       Expects: {base_path}.faiss, {base_path}.meta.json, {base_path}.h5
                       Optional: {base_path}.sequences.json for sequence alignment stats
            use_gpu: Whether to use GPU for FAISS operations.
            metric: Distance metric ('cosine' or 'euclidean').
            verbose: Print progress information.
        """
        if ig is None:
            raise ImportError("igraph is required for clustering. Install with: pip install igraph")
        if leidenalg is None:
            raise ImportError("leidenalg is required for clustering. Install with: pip install leidenalg")
        
        self.base_path = base_path
        self.verbose = verbose
        self.metric = metric
        
        # Load BioVecDB (FAISS index + metadata)
        if self.verbose:
            print(f"[Clusterer] Loading BioVecDB from {base_path}")
        self.db = BioVecDB.load(base_path, use_gpu=use_gpu, metric=metric)
        
        # Load H5 file with embeddings
        if self.verbose:
            print(f"[Clusterer] Loading embeddings from {base_path}.h5")
        h5_data = read_h5(f"{base_path}.h5")
        self.embeddings = h5_data["matrix"].astype(np.float32)
        self.ids = h5_data["ids"]
        self.descriptions = h5_data["descriptions"]
        
        if self.verbose:
            print(f"[Clusterer] Loaded {self.embeddings.shape[0]} embeddings with dim={self.embeddings.shape[1]}")
        
        # Try to load sequences for alignment stats (optional)
        self.sequences: Optional[Dict[str, str]] = None
        try:
            self.sequences = load_sequences(base_path)
            if self.verbose:
                print(f"[Clusterer] Loaded {len(self.sequences)} sequences for alignment")
        except FileNotFoundError:
            if self.verbose:
                print(f"[Clusterer] No sequences.json found (sequence alignment unavailable)")
        
        # Cache for KNN results
        self._knn_cache: Optional[Tuple[int, np.ndarray, np.ndarray]] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.embeddings.shape[0]
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[1]
    
    def search_knn(self, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform KNN search for all embeddings.
        
        Args:
            k: Number of nearest neighbors to find.
            
        Returns:
            Tuple of (distances, indices) arrays with shape (N, k).
            For cosine metric, distances are cosine similarities (higher = more similar).
        """
        # Check cache
        if self._knn_cache is not None and self._knn_cache[0] >= k:
            cached_k, D, I = self._knn_cache
            return D[:, :k], I[:, :k]
        
        if self.verbose:
            print(f"[Clusterer] Searching {k}-NN for {self.n_samples} samples")
        
        D, I = self.db.indexer.search(self.embeddings, k)
        
        # Cache results
        self._knn_cache = (k, D, I)
        
        if self.verbose:
            print(f"[Clusterer] KNN search complete")
        
        return D, I
    
    def build_graph(
        self,
        k: int = 30,
        dist_threshold: float = 0.1,
    ) -> "ig.Graph":
        """
        Build a KNN graph from embeddings.
        
        Args:
            k: Number of nearest neighbors.
            dist_threshold: Maximum cosine distance for edge inclusion.
                           Cosine distance = 1 - cosine_similarity.
                           Lower threshold = stricter connections.
                           
        Returns:
            igraph Graph with weighted edges.
        """
        D, I = self.search_knn(k)
        
        # Convert cosine similarity to cosine distance
        # For cosine metric, D contains similarities (higher = more similar)
        if self.metric == "cosine":
            dist = 1.0 - D
        else:
            dist = D
        
        # Create mask for edges within threshold
        # Exclude self-loops (diagonal) and invalid indices
        mask = (dist < dist_threshold) & (I != -1)
        
        # Remove self-loops (where neighbor index equals row index)
        for i in range(self.n_samples):
            for j in range(k):
                if I[i, j] == i:
                    mask[i, j] = False
        
        if self.verbose:
            n_edges = np.sum(mask)
            print(f"[Clusterer] Building graph with {n_edges} edges (threshold={dist_threshold})")
        
        # Extract edge list
        rows, cols = np.where(mask)
        edges = list(zip(rows.tolist(), I[rows, cols].tolist()))
        
        # Compute edge weights (exponential decay of distance)
        weights = np.exp(-dist[rows, cols]).tolist()
        
        # Build graph
        g = ig.Graph(
            n=self.n_samples,
            edges=edges,
            edge_attrs={"weight": weights},
        )
        
        if self.verbose:
            print(f"[Clusterer] Graph: {g.summary()}")
        
        return g
    
    def cluster(
        self,
        k: int = 30,
        dist_threshold: float = 0.1,
        resolution: float = 0.1,
    ) -> ClusterResult:
        """
        Perform Leiden clustering on the KNN graph.
        
        Args:
            k: Number of nearest neighbors for KNN graph.
            dist_threshold: Maximum cosine distance for edges (0.05-0.5 typical).
            resolution: Leiden resolution parameter.
                       Higher values = more clusters, lower = fewer clusters.
                       
        Returns:
            ClusterResult containing labels and cluster statistics.
        """
        if self.verbose:
            print(f"[Clusterer] Starting clustering: k={k}, dist_threshold={dist_threshold}, resolution={resolution}")
        
        # Build KNN graph
        g = self.build_graph(k=k, dist_threshold=dist_threshold)
        
        if self.verbose:
            print(f"[Clusterer] Running Leiden clustering with resolution={resolution}")
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
        )
        
        labels = np.array(partition.membership)
        n_clusters = len(set(labels))
        modularity = partition.modularity
        
        if self.verbose:
            print(f"[Clusterer] Found {n_clusters} clusters (modularity={modularity:.4f})")
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            ids=self.ids,
            descriptions=self.descriptions,
            modularity=modularity,
        )
    
    def get_cluster_members(self, result: ClusterResult, cluster_id: int) -> Dict:
        """
        Get members of a specific cluster.
        
        Args:
            result: ClusterResult from cluster() method.
            cluster_id: Cluster ID to retrieve.
            
        Returns:
            Dict with 'ids', 'descriptions', and 'indices' of cluster members.
        """
        mask = result.labels == cluster_id
        indices = np.where(mask)[0]
        
        return {
            "cluster_id": cluster_id,
            "size": int(np.sum(mask)),
            "indices": indices,
            "ids": self.ids[mask],
            "descriptions": self.descriptions[mask],
        }
    
    def get_cluster_summary(self, result: ClusterResult) -> List[Dict]:
        """
        Get summary statistics for all clusters.
        
        Args:
            result: ClusterResult from cluster() method.
            
        Returns:
            List of dicts with cluster_id, size, and fraction for each cluster.
        """
        unique, counts = np.unique(result.labels, return_counts=True)
        total = len(result.labels)
        
        summary = []
        for cluster_id, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            summary.append({
                "cluster_id": int(cluster_id),
                "size": int(count),
                "fraction": count / total,
            })
        
        return summary
    
    def compute_cluster_cohesion(
        self,
        result: ClusterResult,
        cluster_id: int,
        max_samples: int = 1000,
        sequence_alignment: bool = True,
    ) -> Dict:
        """
        Compute cohesion statistics for a single cluster.
        
        Args:
            result: ClusterResult from cluster() method.
            cluster_id: Cluster ID to analyze.
            max_samples: Maximum samples for pairwise computation.
            sequence_alignment: If True, also compute pairwise sequence identity stats.
                               Requires sequences.json to be present.
                        
        Returns:
            Dict with: size, mean/min/max/std_similarity, radius, mean_centroid_similarity,
                       and optionally sequence identity stats if sequence_alignment=True.
        """
        mask = result.labels == cluster_id
        indices = np.where(mask)[0]
        
        # Get cluster embeddings
        cluster_embs = self.embeddings[indices]
        
        # Get cluster IDs for sequence lookup
        cluster_ids = None
        if sequence_alignment:
            if self.sequences is None and self.verbose:
                print(f"[Clusterer] Warning: sequence_alignment=True but no sequences loaded")
            cluster_ids = [
                self.ids[idx].decode() if isinstance(self.ids[idx], bytes) else str(self.ids[idx])
                for idx in indices
            ]
        
        return _compute_cluster_cohesion(
            cluster_embeddings=cluster_embs,
            cluster_ids=cluster_ids,
            sequences=self.sequences,
            cluster_id=cluster_id,
            max_samples=max_samples,
            sequence_alignment=sequence_alignment,
        )
    
    def compute_all_cluster_cohesion(
        self,
        result: ClusterResult,
        min_cluster_size: int = 2,
        max_samples: int = 1000,
        sequence_alignment: bool = False,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Compute cohesion statistics for all clusters.
        
        Args:
            result: ClusterResult from cluster() method.
            min_cluster_size: Only compute for clusters with at least this many members.
            max_samples: Maximum samples per cluster for pairwise computation.
            sequence_alignment: If True, also compute pairwise sequence identity stats.
            verbose: Print progress.
            
        Returns:
            List of cohesion dicts sorted by size (descending).
        """
        unique_clusters = np.unique(result.labels)
        cohesion_stats = []
        
        if sequence_alignment and self.sequences is None:
            if verbose:
                print("[Clusterer] Warning: sequence_alignment=True but no sequences loaded")
        
        for i, cluster_id in enumerate(unique_clusters):
            size = np.sum(result.labels == cluster_id)
            if size < min_cluster_size:
                continue
                
            if verbose and (i + 1) % 100 == 0:
                print(f"[Clusterer] Computing cohesion: {i + 1}/{len(unique_clusters)}")
            
            stats = self.compute_cluster_cohesion(result, cluster_id, max_samples, sequence_alignment)
            cohesion_stats.append(stats)
        
        # Sort by size descending
        cohesion_stats.sort(key=lambda x: -x["size"])
        
        if verbose:
            print(f"[Clusterer] Computed cohesion for {len(cohesion_stats)} clusters")
        
        return cohesion_stats
    
    def get_cohesion_summary(self, cohesion_stats: List[Dict]) -> Dict:
        """
        Summarize cohesion statistics across all clusters.
        
        Args:
            cohesion_stats: Output from compute_all_cluster_cohesion().
            
        Returns:
            Dict with aggregate statistics across clusters.
        """
        return compute_cohesion_summary(cohesion_stats)
    
    def get_cluster_representative(self, result: ClusterResult, cluster_id: int) -> Dict:
        """
        Get the representative member of a cluster (closest to centroid).
        
        Args:
            result: ClusterResult from cluster() method.
            cluster_id: Cluster ID to analyze.
            
        Returns:
            Dict with representative id, index, and similarity to centroid.
        """
        mask = result.labels == cluster_id
        indices = np.where(mask)[0]
        
        # Get cluster embeddings and IDs
        cluster_embs = self.embeddings[indices]
        cluster_ids = [
            self.ids[idx].decode() if isinstance(self.ids[idx], bytes) else str(self.ids[idx])
            for idx in indices
        ]
        
        rep_info = find_cluster_representative(cluster_embs, cluster_ids, cluster_id)
        
        # Map local index back to global index
        local_idx = rep_info.get("representative_local_idx")
        global_idx = int(indices[local_idx]) if local_idx is not None and len(indices) > 0 else None
        
        return {
            "cluster_id": cluster_id,
            "representative_id": rep_info["representative_id"],
            "representative_idx": global_idx,
            "centroid_similarity": rep_info["centroid_similarity"],
        }

    def __repr__(self):
        return f"Clusterer(base_path='{self.base_path}', n_samples={self.n_samples}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def cluster_db(
    db_path: str,
    output_path: str,
    k: int = 30,
    dist_threshold: float = 0.1,
    resolution: float = 0.1,
    min_cluster_size: int = 2,
    sequence_alignment: bool = False,
    use_gpu: bool = False,
    verbose: bool = True,
) -> ClusterResult:
    """
    Cluster a BioVecDB and write results to files.
    
    Args:
        db_path: Path to BioVecDB files (without extension).
        output_path: Output path prefix for result files.
        k: Number of nearest neighbors for KNN graph.
        dist_threshold: Maximum cosine distance for edges.
        resolution: Leiden resolution parameter.
        min_cluster_size: Minimum cluster size for cohesion stats.
        sequence_alignment: If True, compute pairwise sequence identity within clusters.
        use_gpu: Whether to use GPU for FAISS.
        verbose: Print progress information.
        
    Output files:
        {output_path}.clusters.tsv - Cluster membership (representative_id, member_id)
        {output_path}.cluster_stats.tsv - Per-cluster statistics
        {output_path}.summary.tsv - Overall summary statistics
        
    Returns:
        ClusterResult object.
    """
    import json
    
    if verbose:
        print(f"[cluster_db] Loading database from {db_path}")
    
    # Initialize clusterer and run clustering
    clusterer = Clusterer(db_path, use_gpu=use_gpu, verbose=verbose)
    result = clusterer.cluster(k=k, dist_threshold=dist_threshold, resolution=resolution)
    
    if verbose:
        print(f"[cluster_db] Computing cluster statistics...")
    
    # Compute cohesion for all clusters
    cohesion_stats = clusterer.compute_all_cluster_cohesion(
        result, min_cluster_size=min_cluster_size, sequence_alignment=sequence_alignment, verbose=verbose
    )
    summary = clusterer.get_cohesion_summary(cohesion_stats)
    
    # Build cluster membership and representative mapping
    if verbose:
        print(f"[cluster_db] Building cluster membership...")
    
    cluster_membership = []  # (representative_id, member_id)
    cluster_stats_rows = []  # Per-cluster stats
    
    unique_clusters = np.unique(result.labels)
    for cluster_id in unique_clusters:
        # Get representative
        rep_info = clusterer.get_cluster_representative(result, cluster_id)
        rep_id = rep_info["representative_id"]
        
        # Get all members
        members = clusterer.get_cluster_members(result, cluster_id)
        for member_id in members["ids"]:
            mid = member_id.decode() if isinstance(member_id, bytes) else str(member_id)
            cluster_membership.append((rep_id, mid, cluster_id))
        
        # Get cohesion stats for this cluster (if computed)
        cohesion = next((c for c in cohesion_stats if c["cluster_id"] == cluster_id), None)
        row = {
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "size": members["size"],
            "mean_similarity": cohesion["mean_similarity"] if cohesion else np.nan,
            "min_similarity": cohesion["min_similarity"] if cohesion else np.nan,
            "max_similarity": cohesion["max_similarity"] if cohesion else np.nan,
            "std_similarity": cohesion["std_similarity"] if cohesion else np.nan,
            "radius": cohesion["radius"] if cohesion else np.nan,
            "mean_centroid_similarity": cohesion["mean_centroid_similarity"] if cohesion else np.nan,
        }
        # Add sequence identity stats if available
        if cohesion and "mean_seq_identity" in cohesion:
            row["mean_seq_identity"] = cohesion["mean_seq_identity"]
            row["min_seq_identity"] = cohesion["min_seq_identity"]
            row["max_seq_identity"] = cohesion["max_seq_identity"]
            row["std_seq_identity"] = cohesion["std_seq_identity"]
        cluster_stats_rows.append(row)
    
    # Write cluster membership TSV
    clusters_file = f"{output_path}.clusters.tsv"
    if verbose:
        print(f"[cluster_db] Writing cluster membership to {clusters_file}")
    with open(clusters_file, "w") as f:
        f.write("representative_id\tmember_id\tcluster_id\n")
        for rep_id, member_id, cluster_id in cluster_membership:
            f.write(f"{rep_id}\t{member_id}\t{cluster_id}\n")
    
    # Write per-cluster stats TSV
    stats_file = f"{output_path}.cluster_stats.tsv"
    if verbose:
        print(f"[cluster_db] Writing cluster stats to {stats_file}")
    with open(stats_file, "w") as f:
        headers = ["cluster_id", "representative_id", "size", "mean_similarity", "min_similarity", 
                   "max_similarity", "std_similarity", "radius", "mean_centroid_similarity"]
        # Add sequence identity headers if available
        if cluster_stats_rows and "mean_seq_identity" in cluster_stats_rows[0]:
            headers.extend(["mean_seq_identity", "min_seq_identity", "max_seq_identity", "std_seq_identity"])
        f.write("\t".join(headers) + "\n")
        for row in sorted(cluster_stats_rows, key=lambda x: -x["size"]):
            values = [str(row.get(h, np.nan)) for h in headers]
            f.write("\t".join(values) + "\n")
    
    # Write summary stats TSV
    summary_file = f"{output_path}.summary.tsv"
    if verbose:
        print(f"[cluster_db] Writing summary to {summary_file}")
    
    # Add clustering parameters to summary
    full_summary = {
        "db_path": db_path,
        "k": k,
        "dist_threshold": dist_threshold,
        "resolution": resolution,
        "n_samples": clusterer.n_samples,
        "n_clusters": result.n_clusters,
        "modularity": result.modularity,
        **summary,
    }
    
    with open(summary_file, "w") as f:
        f.write("parameter\tvalue\n")
        for key, value in full_summary.items():
            f.write(f"{key}\t{value}\n")
    
    if verbose:
        print(f"[cluster_db] Done. Output files:")
        print(f"  - {clusters_file}")
        print(f"  - {stats_file}")
        print(f"  - {summary_file}")
    
    return result


def cluster_sequences(
    fasta_file: str,
    output_path: str,
    embedder_config: Optional[str] = None,
    k: int = 30,
    dist_threshold: float = 0.1,
    resolution: float = 0.1,
    min_cluster_size: int = 2,
    sequence_alignment: bool = False,
    use_gpu: bool = False,
    verbose: bool = True,
) -> ClusterResult:
    """
    Cluster protein sequences from a FASTA file.
    
    This is a convenience function that:
    1. Embeds sequences from the FASTA file
    2. Builds a BioVecDB index
    3. Performs Leiden clustering
    4. Writes output files
    
    Args:
        fasta_file: Path to input FASTA file with protein sequences.
        output_path: Output path prefix for all result files.
        embedder_config: Path to embedder config YAML, or None to use default ESM2.
        k: Number of nearest neighbors for KNN graph.
        dist_threshold: Maximum cosine distance for edges.
        resolution: Leiden resolution parameter.
        min_cluster_size: Minimum cluster size for cohesion stats.
        sequence_alignment: If True, compute pairwise sequence identity within clusters.
        use_gpu: Whether to use GPU for embedding and FAISS.
        verbose: Print progress information.
        
    Output files:
        {output_path}.faiss - FAISS index
        {output_path}.meta.json - Metadata JSON
        {output_path}.h5 - Embeddings HDF5
        {output_path}.sequences.json - ID to sequence mapping
        {output_path}.fasta - Copy of input FASTA
        {output_path}.clusters.tsv - Cluster membership
        {output_path}.cluster_stats.tsv - Per-cluster statistics
        {output_path}.summary.tsv - Overall summary
        
    Returns:
        ClusterResult object.
        
    Example:
        >>> from biovec.clusterer import cluster_sequences
        >>> result = cluster_sequences(
        ...     fasta_file="proteins.fasta",
        ...     output_path="output/proteins",
        ...     dist_threshold=0.1,
        ...     resolution=0.1,
        ...     sequence_alignment=True,  # Compute pairwise sequence identity
        ... )
        >>> print(f"Found {result.n_clusters} clusters")
    """
    if verbose:
        print(f"[cluster_sequences] Input FASTA: {fasta_file}")
        print(f"[cluster_sequences] Output prefix: {output_path}")
    
    # Step 1: Build database (embed sequences and create FAISS index)
    if verbose:
        print(f"[cluster_sequences] Building database...")
    
    db = build_db(
        fasta_file=fasta_file,
        embedder_config=embedder_config,
        metric="cosine",
        save_path=output_path,
    )
    
    if verbose:
        print(f"[cluster_sequences] Database built: {db.ntotal} sequences, dim={db.dim}")
    
    # Step 2: Cluster the database
    result = cluster_db(
        db_path=output_path,
        output_path=output_path,
        k=k,
        dist_threshold=dist_threshold,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        sequence_alignment=sequence_alignment,
        use_gpu=use_gpu,
        verbose=verbose,
    )
    
    if verbose:
        print(f"[cluster_sequences] Complete! Found {result.n_clusters} clusters")
    
    return result

