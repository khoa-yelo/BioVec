#!/usr/bin/env python
"""
Test script for the Clusterer class and clustering functions.

Usage:
    python tests/test_clusterer.py                              # Test Clusterer class
    python tests/test_clusterer.py --sequence-alignment         # With sequence identity stats
    python tests/test_clusterer.py --sweep                      # Parameter sweep
    python tests/test_clusterer.py --cluster-db                 # Test cluster_db function
    python tests/test_clusterer.py --cluster-fasta              # Test cluster_sequences function
    python tests/test_clusterer.py --cluster-db --sequence-alignment  # With seq identity

Requires a pre-built BioVecDB at test_data/test_outs/bacillus with:
    - bacillus.faiss
    - bacillus.meta.json
    - bacillus.h5
    - bacillus.sequences.json (optional, for --sequence-alignment)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biovec.clusterer import Clusterer, cluster_db, cluster_sequences


def test_clusterer(
    base_path: str,
    dist_threshold: float = 0.1,
    resolution: float = 0.1,
    k: int = 30,
    use_gpu: bool = False,
    sequence_alignment: bool = False,
):
    """
    Test clustering on a BioVecDB.
    
    Args:
        base_path: Path to BioVecDB files (without extension).
        dist_threshold: Max cosine distance for edges.
        resolution: Leiden resolution parameter.
        k: Number of nearest neighbors.
        use_gpu: Whether to use GPU for FAISS.
        sequence_alignment: Whether to compute sequence identity stats.
    """
    print("=" * 60)
    print("CLUSTERER TEST")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"Parameters: k={k}, dist_threshold={dist_threshold}, resolution={resolution}")
    print(f"Sequence alignment: {sequence_alignment}")
    print()

    # Initialize clusterer
    print("Loading BioVecDB and embeddings...")
    clusterer = Clusterer(base_path, use_gpu=use_gpu, verbose=False)
    print(f"  {clusterer}")
    if clusterer.sequences:
        print(f"  Sequences loaded: {len(clusterer.sequences)}")
    print()

    # Run clustering
    print("Running Leiden clustering...")
    result = clusterer.cluster(k=k, dist_threshold=dist_threshold, resolution=resolution)
    print(f"  Found {result.n_clusters} clusters")
    print(f"  Modularity: {result.modularity:.4f}")
    print()

    # Compute cohesion statistics
    print("Computing cluster cohesion statistics...")
    cohesion_stats = clusterer.compute_all_cluster_cohesion(
        result, min_cluster_size=2, sequence_alignment=sequence_alignment, verbose=False
    )
    summary = clusterer.get_cohesion_summary(cohesion_stats)

    print()
    print("=" * 60)
    print("OVERALL COHESION SUMMARY")
    print("=" * 60)
    print(f"  Clusters analyzed (size >= 2): {summary['n_clusters']}")
    print(f"  Total samples in clusters:     {summary['total_samples']}")
    print(f"  Mean cluster size:             {summary['mean_cluster_size']:.1f}")
    print(f"  Median cluster size:           {summary['median_cluster_size']:.1f}")
    print()
    print(f"  Weighted mean similarity:      {summary['weighted_mean_similarity']:.4f}")
    print()
    print(f"  Mean similarity:  avg={summary['avg_mean_similarity']:.4f}  min={summary['min_mean_similarity']:.4f}  max={summary['max_mean_similarity']:.4f}")
    print(f"  Min similarity:   avg={summary['avg_min_similarity']:.4f}  min={summary['min_min_similarity']:.4f}  max={summary['max_min_similarity']:.4f}")
    print(f"  Radius:           avg={summary['avg_radius']:.4f}  min={summary['min_radius']:.4f}  max={summary['max_radius']:.4f}")
    
    # Show sequence identity stats if available
    if sequence_alignment and 'avg_mean_seq_identity' in summary:
        print()
        print("  --- Sequence Identity (%) ---")
        print(f"  Weighted mean seq identity:    {summary['weighted_mean_seq_identity']:.1f}%")
        print(f"  Mean seq identity: avg={summary['avg_mean_seq_identity']:.1f}%  min={summary['min_mean_seq_identity']:.1f}%  max={summary['max_mean_seq_identity']:.1f}%")
        print(f"  Min seq identity:  avg={summary['avg_min_seq_identity']:.1f}%  min={summary['min_min_seq_identity']:.1f}%  max={summary['max_min_seq_identity']:.1f}%")
    print()

    # Top clusters by size
    print("=" * 60)
    print("TOP 10 CLUSTERS BY SIZE")
    print("=" * 60)
    has_seq_id = cohesion_stats and "mean_seq_identity" in cohesion_stats[0]
    if has_seq_id:
        print(f"{'ID':>6} {'Size':>6} {'Mean Sim':>10} {'Min Sim':>10} {'Radius':>8} {'MeanSeqId':>10}")
        print("-" * 62)
    else:
        print(f"{'ID':>6} {'Size':>6} {'Mean Sim':>10} {'Min Sim':>10} {'Radius':>8}")
        print("-" * 50)
    for c in cohesion_stats[:10]:
        line = (
            f"{c['cluster_id']:>6} "
            f"{c['size']:>6} "
            f"{c['mean_similarity']:>10.4f} "
            f"{c['min_similarity']:>10.4f} "
            f"{c['radius']:>8.4f}"
        )
        if has_seq_id:
            seq_id = c.get('mean_seq_identity', float('nan'))
            line += f" {seq_id:>9.1f}%"
        print(line)
    print()

    # Tightest clusters
    print("=" * 60)
    print("TOP 10 TIGHTEST CLUSTERS (size >= 5)")
    print("=" * 60)
    tight = sorted(
        [c for c in cohesion_stats if c["size"] >= 5],
        key=lambda x: -x["mean_similarity"],
    )[:10]
    
    if tight:
        if has_seq_id:
            print(f"{'ID':>6} {'Size':>6} {'Mean Sim':>10} {'Min Sim':>10} {'Radius':>8} {'MeanSeqId':>10}")
            print("-" * 62)
        else:
            print(f"{'ID':>6} {'Size':>6} {'Mean Sim':>10} {'Min Sim':>10} {'Radius':>8}")
            print("-" * 50)
        for c in tight:
            line = (
                f"{c['cluster_id']:>6} "
                f"{c['size']:>6} "
                f"{c['mean_similarity']:>10.4f} "
                f"{c['min_similarity']:>10.4f} "
                f"{c['radius']:>8.4f}"
            )
            if has_seq_id:
                seq_id = c.get('mean_seq_identity', float('nan'))
                line += f" {seq_id:>9.1f}%"
            print(line)
    else:
        print("  No clusters with size >= 5 found.")
    print()

    # Show members of the largest cluster
    print("=" * 60)
    print("SAMPLE MEMBERS FROM LARGEST CLUSTER")
    print("=" * 60)
    if cohesion_stats:
        largest_id = cohesion_stats[0]["cluster_id"]
        members = clusterer.get_cluster_members(result, largest_id)
        print(f"  Cluster ID: {largest_id}")
        print(f"  Size: {members['size']}")
        print(f"  Sample IDs:")
        for pid in members["ids"][:10]:
            # Handle byte strings from h5py
            pid_str = pid.decode() if isinstance(pid, bytes) else pid
            print(f"    - {pid_str}")
    print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return clusterer, result, cohesion_stats


def test_parameter_sweep(base_path: str, use_gpu: bool = False, sequence_alignment: bool = False):
    """
    Test different clustering parameters with cohesion statistics.
    
    Args:
        base_path: Path to BioVecDB files (without extension).
        use_gpu: Whether to use GPU for FAISS.
        sequence_alignment: Whether to compute sequence identity stats.
    """
    print("=" * 140)
    print("PARAMETER SWEEP WITH COHESION STATISTICS")
    print("=" * 140)
    print()

    clusterer = Clusterer(base_path, use_gpu=use_gpu, verbose=False)
    print(f"Loaded: {clusterer}")
    if clusterer.sequences:
        print(f"Sequences loaded: {len(clusterer.sequences)}")
    print(f"Sequence alignment: {sequence_alignment}")
    print()

    # Header - add sequence identity columns if enabled
    header = (
        f"{'dist':>6} {'res':>6} {'#Clust':>8} {'Largest':>8} {'Modul':>7} | "
        f"{'AvgMeanSim':>11} {'MinMeanSim':>11} {'AvgMinSim':>10} {'MinMinSim':>10} {'MaxRadius':>10}"
    )
    if sequence_alignment:
        header += f" | {'AvgSeqId':>9} {'MinSeqId':>9} {'MaxSeqId':>9}"
    print(header)
    print("-" * (140 if sequence_alignment else 110))

    for dist_thresh in [0.01, 0.02, 0.05, 0.1]:
        for resolution in [0.01, 0.05, 0.1]:
            result = clusterer.cluster(k=30, dist_threshold=dist_thresh, resolution=resolution)
            cluster_summary = clusterer.get_cluster_summary(result)
            largest = cluster_summary[0]["size"] if cluster_summary else 0
            
            # Compute cohesion statistics
            cohesion_stats = clusterer.compute_all_cluster_cohesion(
                result, min_cluster_size=2, sequence_alignment=sequence_alignment, verbose=False
            )
            cs = clusterer.get_cohesion_summary(cohesion_stats)
            
            # Extract cohesion metrics (handle empty case)
            if cs:
                line = (
                    f"{dist_thresh:>6.2f} {resolution:>6.2f} "
                    f"{result.n_clusters:>8} {largest:>8} {result.modularity:>7.3f} | "
                    f"{cs['avg_mean_similarity']:>11.4f} {cs['min_mean_similarity']:>11.4f} "
                    f"{cs['avg_min_similarity']:>10.4f} {cs['min_min_similarity']:>10.4f} {cs['max_radius']:>10.4f}"
                )
                if sequence_alignment and 'avg_mean_seq_identity' in cs:
                    line += (
                        f" | {cs['avg_mean_seq_identity']:>8.1f}% "
                        f"{cs['min_mean_seq_identity']:>8.1f}% "
                        f"{cs['max_mean_seq_identity']:>8.1f}%"
                    )
                print(line)
            else:
                line = (
                    f"{dist_thresh:>6.2f} {resolution:>6.2f} "
                    f"{result.n_clusters:>8} {largest:>8} {result.modularity:>7.3f} | "
                    f"{'nan':>11} {'nan':>11} {'nan':>10} {'nan':>10} {'nan':>10}"
                )
                if sequence_alignment:
                    line += f" | {'nan':>9} {'nan':>9} {'nan':>9}"
                print(line)
        print()  # Blank line between dist_thresh groups
    
    print()
    print("Legend:")
    print("  dist       = cosine distance threshold (lower = stricter)")
    print("  res        = Leiden resolution (higher = more clusters)")
    print("  #Clust     = number of clusters")
    print("  Largest    = size of largest cluster")
    print("  Modul      = modularity score")
    print("  AvgMeanSim = average of mean pairwise similarities across clusters")
    print("  MinMeanSim = minimum mean pairwise similarity (worst cluster cohesion)")
    print("  AvgMinSim  = average of min pairwise similarities (avg cluster diameter)")
    print("  MinMinSim  = minimum of min similarities (worst-case pair in any cluster)")
    print("  MaxRadius  = maximum radius across all clusters")
    if sequence_alignment:
        print("  AvgSeqId   = average mean sequence identity across clusters (%)")
        print("  MinSeqId   = minimum mean sequence identity (worst cluster)")
        print("  MaxSeqId   = maximum mean sequence identity (best cluster)")
    print()


def test_cluster_db(
    base_path: str,
    output_path: str,
    dist_threshold: float = 0.1,
    resolution: float = 0.1,
    k: int = 30,
    use_gpu: bool = False,
    sequence_alignment: bool = False,
):
    """
    Test cluster_db function that writes output files.
    
    Args:
        base_path: Path to BioVecDB files (without extension).
        output_path: Output path prefix for result files.
        dist_threshold: Max cosine distance for edges.
        resolution: Leiden resolution parameter.
        k: Number of nearest neighbors.
        use_gpu: Whether to use GPU for FAISS.
        sequence_alignment: Whether to compute sequence identity stats.
    """
    print("=" * 60)
    print("TEST cluster_db FUNCTION")
    print("=" * 60)
    print(f"Input DB: {base_path}")
    print(f"Output prefix: {output_path}")
    print(f"Parameters: k={k}, dist_threshold={dist_threshold}, resolution={resolution}")
    print(f"Sequence alignment: {sequence_alignment}")
    print()
    
    # Run cluster_db
    result = cluster_db(
        db_path=base_path,
        output_path=output_path,
        k=k,
        dist_threshold=dist_threshold,
        resolution=resolution,
        sequence_alignment=sequence_alignment,
        use_gpu=use_gpu,
        verbose=True,
    )
    
    # Verify output files exist
    print()
    print("=" * 60)
    print("VERIFYING OUTPUT FILES")
    print("=" * 60)
    
    output_files = [
        f"{output_path}.clusters.tsv",
        f"{output_path}.cluster_stats.tsv",
        f"{output_path}.summary.tsv",
    ]
    
    for fpath in output_files:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            with open(fpath) as f:
                lines = len(f.readlines())
            print(f"  ✓ {fpath} ({size:,} bytes, {lines} lines)")
        else:
            print(f"  ✗ {fpath} - NOT FOUND")
    
    # Parse and display summary stats
    print()
    print("=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    summary_data = {}
    with open(f"{output_path}.summary.tsv") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                summary_data[parts[0]] = parts[1]
    
    print(f"  Total samples:      {summary_data.get('n_samples', 'N/A')}")
    print(f"  Number of clusters: {summary_data.get('n_clusters', 'N/A')}")
    print(f"  Modularity:         {summary_data.get('modularity', 'N/A')}")
    print()
    print(f"  Mean cluster size:  {float(summary_data.get('mean_cluster_size', 0)):.1f}")
    print(f"  Weighted mean sim:  {float(summary_data.get('weighted_mean_similarity', 0)):.4f}")
    
    # Show sequence identity stats if available
    if 'avg_mean_seq_identity' in summary_data:
        print()
        print("  --- Sequence Identity (%) ---")
        print(f"  Weighted mean seq identity: {float(summary_data.get('weighted_mean_seq_identity', 0)):.1f}%")
        print(f"  Mean seq identity:  avg={float(summary_data.get('avg_mean_seq_identity', 0)):.1f}%  "
              f"min={float(summary_data.get('min_mean_seq_identity', 0)):.1f}%  "
              f"max={float(summary_data.get('max_mean_seq_identity', 0)):.1f}%")
        print(f"  Min seq identity:   avg={float(summary_data.get('avg_min_seq_identity', 0)):.1f}%  "
              f"min={float(summary_data.get('min_min_seq_identity', 0)):.1f}%  "
              f"max={float(summary_data.get('max_min_seq_identity', 0)):.1f}%")
    
    # Show sample of each output file
    print()
    print("=" * 60)
    print("SAMPLE OUTPUT: clusters.tsv (first 10 lines)")
    print("=" * 60)
    with open(f"{output_path}.clusters.tsv") as f:
        for i, line in enumerate(f):
            if i >= 10:
                print("  ...")
                break
            print(f"  {line.rstrip()}")
    
    print()
    print("=" * 60)
    print("SAMPLE OUTPUT: cluster_stats.tsv (first 10 lines)")
    print("=" * 60)
    with open(f"{output_path}.cluster_stats.tsv") as f:
        for i, line in enumerate(f):
            if i >= 10:
                print("  ...")
                break
            print(f"  {line.rstrip()}")
    
    print()
    print("=" * 60)
    print("SAMPLE OUTPUT: summary.tsv")
    print("=" * 60)
    with open(f"{output_path}.summary.tsv") as f:
        for line in f:
            print(f"  {line.rstrip()}")
    
    print()
    print("=" * 60)
    print("TEST cluster_db COMPLETE")
    print("=" * 60)
    
    return result


def test_cluster_sequences(
    fasta_file: str,
    output_path: str,
    embedder_config: str = None,
    dist_threshold: float = 0.1,
    resolution: float = 0.1,
    k: int = 30,
    use_gpu: bool = False,
    sequence_alignment: bool = False,
):
    """
    Test cluster_sequences function (end-to-end: FASTA -> clusters).
    
    Args:
        fasta_file: Path to input FASTA file.
        output_path: Output path prefix for result files.
        embedder_config: Path to embedder config YAML.
        dist_threshold: Max cosine distance for edges.
        resolution: Leiden resolution parameter.
        k: Number of nearest neighbors.
        use_gpu: Whether to use GPU.
        sequence_alignment: Whether to compute sequence identity stats.
    """
    print("=" * 60)
    print("TEST cluster_sequences FUNCTION")
    print("=" * 60)
    print(f"Input FASTA: {fasta_file}")
    print(f"Output prefix: {output_path}")
    print(f"Embedder config: {embedder_config or 'default (ESM2)'}")
    print(f"Parameters: k={k}, dist_threshold={dist_threshold}, resolution={resolution}")
    print(f"Sequence alignment: {sequence_alignment}")
    print()
    
    # Run cluster_sequences
    result = cluster_sequences(
        fasta_file=fasta_file,
        output_path=output_path,
        embedder_config=embedder_config,
        k=k,
        dist_threshold=dist_threshold,
        resolution=resolution,
        sequence_alignment=sequence_alignment,
        use_gpu=use_gpu,
        verbose=True,
    )
    
    # Verify output files exist
    print()
    print("=" * 60)
    print("VERIFYING OUTPUT FILES")
    print("=" * 60)
    
    output_files = [
        f"{output_path}.faiss",
        f"{output_path}.meta.json",
        f"{output_path}.h5",
        f"{output_path}.sequences.json",
        f"{output_path}.fasta",
        f"{output_path}.clusters.tsv",
        f"{output_path}.cluster_stats.tsv",
        f"{output_path}.summary.tsv",
    ]
    
    for fpath in output_files:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  ✓ {fpath} ({size:,} bytes)")
        else:
            print(f"  ✗ {fpath} - NOT FOUND")
    
    # Parse and display summary stats
    print()
    print("=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    summary_data = {}
    with open(f"{output_path}.summary.tsv") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                summary_data[parts[0]] = parts[1]
    
    print(f"  Total samples:      {summary_data.get('n_samples', 'N/A')}")
    print(f"  Number of clusters: {summary_data.get('n_clusters', 'N/A')}")
    print(f"  Modularity:         {summary_data.get('modularity', 'N/A')}")
    print()
    print(f"  Mean cluster size:  {float(summary_data.get('mean_cluster_size', 0)):.1f}")
    print(f"  Weighted mean sim:  {float(summary_data.get('weighted_mean_similarity', 0)):.4f}")
    
    # Show sequence identity stats if available
    if 'avg_mean_seq_identity' in summary_data:
        print()
        print("  --- Sequence Identity (%) ---")
        print(f"  Weighted mean seq identity: {float(summary_data.get('weighted_mean_seq_identity', 0)):.1f}%")
        print(f"  Mean seq identity:  avg={float(summary_data.get('avg_mean_seq_identity', 0)):.1f}%  "
              f"min={float(summary_data.get('min_mean_seq_identity', 0)):.1f}%  "
              f"max={float(summary_data.get('max_mean_seq_identity', 0)):.1f}%")
        print(f"  Min seq identity:   avg={float(summary_data.get('avg_min_seq_identity', 0)):.1f}%  "
              f"min={float(summary_data.get('min_min_seq_identity', 0)):.1f}%  "
              f"max={float(summary_data.get('max_min_seq_identity', 0)):.1f}%")
    
    print()
    print("=" * 60)
    print("TEST cluster_sequences COMPLETE")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Clusterer on BioVecDB")
    parser.add_argument(
        "--base-path",
        default="test_data/test_outs/bacillus",
        help="Path to BioVecDB files (without extension)",
    )
    parser.add_argument(
        "--fasta",
        default="test_data/bacillus_subtilis_py79.faa",
        help="Path to FASTA file for cluster_sequences test",
    )
    parser.add_argument(
        "--output",
        default="test_data/test_outs/cluster_test",
        help="Output path prefix for cluster_db/cluster_sequences tests",
    )
    parser.add_argument(
        "--embedder-config",
        default=None,
        help="Path to embedder config YAML for cluster_sequences",
    )
    parser.add_argument("--dist-threshold", type=float, default=0.1, help="Cosine distance threshold")
    parser.add_argument("--resolution", type=float, default=0.1, help="Leiden resolution")
    parser.add_argument("--k", type=int, default=30, help="Number of nearest neighbors")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for FAISS/embedding")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--cluster-db", action="store_true", help="Test cluster_db function")
    parser.add_argument("--cluster-fasta", action="store_true", help="Test cluster_sequences function")
    parser.add_argument("--sequence-alignment", action="store_true", help="Compute pairwise sequence identity within clusters")
    args = parser.parse_args()

    if args.sweep:
        test_parameter_sweep(args.base_path, use_gpu=args.gpu, sequence_alignment=args.sequence_alignment)
    elif args.cluster_db:
        test_cluster_db(
            args.base_path,
            args.output,
            dist_threshold=args.dist_threshold,
            resolution=args.resolution,
            k=args.k,
            use_gpu=args.gpu,
            sequence_alignment=args.sequence_alignment,
        )
    elif args.cluster_fasta:
        test_cluster_sequences(
            args.fasta,
            args.output,
            embedder_config=args.embedder_config,
            dist_threshold=args.dist_threshold,
            resolution=args.resolution,
            k=args.k,
            use_gpu=args.gpu,
            sequence_alignment=args.sequence_alignment,
        )
    else:
        test_clusterer(
            args.base_path,
            dist_threshold=args.dist_threshold,
            resolution=args.resolution,
            k=args.k,
            use_gpu=args.gpu,
            sequence_alignment=args.sequence_alignment,
        )
