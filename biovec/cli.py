#!/usr/bin/env python
"""BioVec CLI — embed, index, and search protein sequences."""

import argparse
import sys


def cmd_build_db(args):
    """Build a protein embedding database from FASTA."""
    from .dbbuilder.biovecdb import build_db
    
    db = build_db(
        fasta_file=args.fasta,
        embedder_config=args.embedder_config,
        metric=args.metric,
        save_path=args.output,
    )
    print(f"[biovec] Built database with {db.ntotal:,} sequences → {args.output}.*")


def cmd_search_db(args):
    """Search a database with query sequences."""
    from .dbbuilder.biovecdb import load_db, search_db
    
    db = load_db(args.database, metric=args.metric)
    print(f"[biovec] Loaded database: {db.ntotal:,} sequences, dim={db.dim}")
    
    results = search_db(db, query_fasta=args.query, top_k=args.top_k)
    
    for res in results:
        print(f"\n=== Query: {res['query_id']} ===")
        for hit in res["hits"]:
            desc = hit.get("description", "")
            desc_str = f" | {desc[:50]}..." if desc and len(desc) > 50 else f" | {desc}" if desc else ""
            print(f"  {hit['rank']:>2}. {hit['id']}  dist={hit['distance']:.4f}{desc_str}")


def cmd_embed(args):
    """Embed protein sequences from FASTA to H5."""
    from .embedders.protein_embedder import embed_protein
    
    out = embed_protein(
        embedder_config=args.embedder_config,
        fasta_file=args.fasta,
        save_h5=args.output,
        verbose=True,
    )
    print(f"[biovec] Embedded {len(out['ids']):,} sequences → {args.output}")


def cmd_gene_call(args):
    """Predict genes from nucleotide FASTA."""
    from .genecaller.pyrodigal_call import run_pyrodigal
    
    run_pyrodigal(
        fasta_path=args.fasta,
        output_dir=args.output,
        mode=args.mode,
        threads=args.threads,
    )


def cmd_csv_to_fasta(args):
    """Convert CSV to FASTA format."""
    from .io.fasta import csv_to_fasta
    
    count = csv_to_fasta(
        csv_path=args.csv,
        fasta_path=args.output,
        id_col=args.id_col,
        seq_col=args.seq_col,
        desc_col=args.desc_col,
        delimiter=args.delimiter,
        has_header=not args.no_header,
    )
    print(f"[biovec] Converted {count:,} sequences → {args.output}")


def cmd_cluster(args):
    """Cluster a BioVecDB or FASTA file."""
    from .clusterer import cluster_db, cluster_sequences

    verbose = not args.quiet

    if args.fasta:
        result = cluster_sequences(
            fasta_file=args.fasta,
            output_path=args.output,
            embedder_config=args.embedder_config,
            k=args.k,
            dist_threshold=args.dist_threshold,
            resolution=args.resolution,
            min_cluster_size=args.min_cluster_size,
            sequence_alignment=args.sequence_alignment,
            method=args.method,
            mst_cut_threshold=args.mst_cut_threshold,
            use_gpu=args.use_gpu,
            verbose=verbose,
        )
    else:
        result = cluster_db(
            db_path=args.database,
            output_path=args.output,
            k=args.k,
            dist_threshold=args.dist_threshold,
            resolution=args.resolution,
            min_cluster_size=args.min_cluster_size,
            sequence_alignment=args.sequence_alignment,
            method=args.method,
            mst_cut_threshold=args.mst_cut_threshold,
            use_gpu=args.use_gpu,
            verbose=verbose,
        )

    if verbose:
        print(f"[biovec] {result.n_clusters} clusters from {len(result.labels)} sequences "
              f"(modularity={result.modularity:.4f})")
        print(f"[biovec] Output → {args.output}.*")


def cmd_find_peak(args):
    """Detect peaks in a score column of a CSV/TSV file."""
    from .peakfinder import find_peak_in_csv

    intervals, df = find_peak_in_csv(
        csv_path=args.input,
        score_col=args.score_col,
        output_path=args.output,
        smooth_size=args.smooth_size,
        prominence=args.prominence,
        distance=args.distance,
        rel_height=args.rel_height,
        flip_score=not args.no_flip,
        verbose=not args.quiet,
    )

    if not args.quiet:
        n_det = int(df["detected"].sum())
        print(f"[biovec] {len(intervals)} interval(s), {n_det}/{len(df)} rows detected")
        for i, (s, e) in enumerate(intervals):
            print(f"  interval {i}: rows {s}–{e} (span {e - s + 1})")
        if args.output:
            print(f"[biovec] Output → {args.output}")


def cmd_remote_score(args):
    """Score query proteins against a database and output TSV."""
    from .peakfinder import remote_score
    
    # Determine input type
    fasta = args.fasta if args.fasta else None
    embeddings_h5 = args.embeddings if args.embeddings else None
    query_db = args.query_db if args.query_db else None
    
    n_scored = remote_score(
        db_path=args.database,
        output_path=args.output,
        fasta=fasta,
        embeddings_h5=embeddings_h5,
        query_db=query_db,
        use_gpu=args.use_gpu,
        metric=args.metric,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(f"[biovec] Scored {n_scored:,} queries → {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="biovec",
        description="BioVec — Protein embedding and semantic search toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # build-db
    # -------------------------------------------------------------------------
    p_build = subparsers.add_parser("build-db", help="Build embedding database from FASTA")
    p_build.add_argument("--fasta", "-f", required=True, help="Input FASTA file")
    p_build.add_argument("--output", "-o", required=True, help="Output database base path")
    p_build.add_argument(
        "--embedder-config", "-e",
        default="glm2",
        help="Embedder config: template name (esm2, t5, glm2) or YAML path",
    )
    p_build.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p_build.set_defaults(func=cmd_build_db)

    # -------------------------------------------------------------------------
    # search-db
    # -------------------------------------------------------------------------
    p_search = subparsers.add_parser("search-db", help="Search database with query FASTA")
    p_search.add_argument("--database", "-d", required=True, help="Database base path")
    p_search.add_argument("--query", "-q", required=True, help="Query FASTA file")
    p_search.add_argument("--top-k", "-k", type=int, default=10, help="Number of hits")
    p_search.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p_search.set_defaults(func=cmd_search_db)

    # -------------------------------------------------------------------------
    # embed
    # -------------------------------------------------------------------------
    p_embed = subparsers.add_parser("embed", help="Embed FASTA sequences to H5")
    p_embed.add_argument("--fasta", "-f", required=True, help="Input FASTA file")
    p_embed.add_argument("--output", "-o", required=True, help="Output H5 file")
    p_embed.add_argument(
        "--embedder-config", "-e",
        default="glm2",
        help="Embedder config: template name or YAML path",
    )
    p_embed.set_defaults(func=cmd_embed)

    # -------------------------------------------------------------------------
    # gene-call
    # -------------------------------------------------------------------------
    p_gene = subparsers.add_parser("gene-call", help="Predict genes from nucleotide FASTA")
    p_gene.add_argument("--fasta", "-f", required=True, help="Input nucleotide FASTA")
    p_gene.add_argument("--output", "-o", required=True, help="Output directory")
    p_gene.add_argument(
        "--mode", "-m",
        choices=["bacteria", "virus", "metagenome"],
        default="metagenome",
        help="Prediction mode",
    )
    p_gene.add_argument("--threads", "-t", type=int, default=1)
    p_gene.set_defaults(func=cmd_gene_call)

    # -------------------------------------------------------------------------
    # csv-to-fasta
    # -------------------------------------------------------------------------
    p_csv = subparsers.add_parser("csv-to-fasta", help="Convert CSV to FASTA")
    p_csv.add_argument("--csv", "-c", required=True, help="Input CSV file")
    p_csv.add_argument("--output", "-o", required=True, help="Output FASTA file")
    p_csv.add_argument("--id-col", required=True, help="Column for sequence ID")
    p_csv.add_argument("--seq-col", required=True, help="Column for sequence")
    p_csv.add_argument("--desc-col", default=None, help="Column for description")
    p_csv.add_argument("--delimiter", default=",", help="CSV delimiter")
    p_csv.add_argument("--no-header", action="store_true", help="CSV has no header row")
    p_csv.set_defaults(func=cmd_csv_to_fasta)

    # -------------------------------------------------------------------------
    # cluster
    # -------------------------------------------------------------------------
    p_clust = subparsers.add_parser(
        "cluster",
        help="Cluster proteins using Leiden or MST algorithm",
        description="Cluster a BioVecDB or FASTA file using KNN graph + Leiden/MST. "
                    "Writes clusters.tsv, cluster_stats.tsv, and summary.tsv.",
    )
    # Input (mutually exclusive: existing DB or raw FASTA)
    clust_input = p_clust.add_mutually_exclusive_group(required=True)
    clust_input.add_argument(
        "--database", "-d",
        help="Existing BioVecDB base path (without extension)",
    )
    clust_input.add_argument(
        "--fasta", "-f",
        help="Input FASTA file (will build DB first, then cluster)",
    )
    # Output
    p_clust.add_argument(
        "--output", "-o",
        required=True,
        help="Output path prefix for result files",
    )
    # Clustering method
    p_clust.add_argument(
        "--method",
        choices=["leiden", "mst"],
        default="leiden",
        help="Clustering method (default: leiden)",
    )
    # KNN parameters
    p_clust.add_argument("--k", type=int, default=30, help="Number of nearest neighbors (default: 30)")
    p_clust.add_argument(
        "--dist-threshold",
        type=float,
        default=0.1,
        help="Max cosine distance for KNN edges (default: 0.1)",
    )
    # Leiden parameters
    p_clust.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Leiden resolution parameter (default: 0.1)",
    )
    # MST parameters
    p_clust.add_argument(
        "--mst-cut-threshold",
        type=float,
        default=0.1,
        help="MST edge cut threshold, only for method=mst (default: 0.1)",
    )
    # Stats options
    p_clust.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Min cluster size for cohesion stats (default: 2)",
    )
    p_clust.add_argument(
        "--sequence-alignment",
        action="store_true",
        help="Compute pairwise sequence identity within clusters",
    )
    # Embedder (only for --fasta mode)
    p_clust.add_argument(
        "--embedder-config", "-e",
        default=None,
        help="Embedder config (glm2, esm2, t5, or YAML path). Only used with --fasta.",
    )
    # Runtime
    p_clust.add_argument("--use-gpu", action="store_true", help="Use GPU for FAISS operations")
    p_clust.add_argument("--quiet", "-Q", action="store_true", help="Suppress progress output")
    p_clust.set_defaults(func=cmd_cluster)

    # -------------------------------------------------------------------------
    # find-peak
    # -------------------------------------------------------------------------
    p_peak = subparsers.add_parser(
        "find-peak",
        help="Detect peaks/intervals in a score column of a CSV/TSV",
        description="Run peak detection on a score column from a CSV/TSV file. "
                    "Adds a 'detected' column (1 inside interval, 0 outside) and "
                    "reports the detected intervals.",
    )
    p_peak.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV or TSV file",
    )
    p_peak.add_argument(
        "--score-col", "-s",
        required=True,
        help="Column name containing scores to analyse",
    )
    p_peak.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV/TSV with added 'detected' column (default: print only)",
    )
    p_peak.add_argument(
        "--smooth-size",
        type=int,
        default=20,
        help="Gaussian smoothing sigma (default: 20)",
    )
    p_peak.add_argument(
        "--prominence",
        type=float,
        default=0.1,
        help="Peak prominence threshold (default: 0.1)",
    )
    p_peak.add_argument(
        "--distance",
        type=int,
        default=5,
        help="Minimum distance between peaks (default: 5)",
    )
    p_peak.add_argument(
        "--rel-height",
        type=float,
        default=0.15,
        help="Relative height for peak width calculation (default: 0.15)",
    )
    p_peak.add_argument(
        "--no-flip",
        action="store_true",
        help="Do NOT flip scores (by default scores are flipped to detect valleys)",
    )
    p_peak.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress progress output",
    )
    p_peak.set_defaults(func=cmd_find_peak)

    # -------------------------------------------------------------------------
    # remote-score
    # -------------------------------------------------------------------------
    p_rscore = subparsers.add_parser(
        "remote-score",
        help="Score query proteins against a database",
        description="Score query proteins against a BioVecDB and output TSV with "
                    "nearest neighbor information (query_id, score, neighbor_id, neighbor_description).",
    )
    p_rscore.add_argument(
        "--database", "-d",
        required=True,
        help="Target database base path (BioVecDB without extension)",
    )
    p_rscore.add_argument(
        "--output", "-o",
        required=True,
        help="Output TSV file path",
    )
    # Query input (mutually exclusive)
    query_group = p_rscore.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--fasta", "-f",
        help="Query FASTA file (will embed using database's embedder config)",
    )
    query_group.add_argument(
        "--embeddings", "-e",
        help="Query embeddings H5 file (must have 'matrix' dataset)",
    )
    query_group.add_argument(
        "--query-db", "-q",
        help="Query BioVecDB base path (without extension)",
    )
    # Options
    p_rscore.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric (default: cosine)",
    )
    p_rscore.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for FAISS operations",
    )
    p_rscore.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress progress output",
    )
    p_rscore.set_defaults(func=cmd_remote_score)

    # -------------------------------------------------------------------------
    # Parse and run
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
