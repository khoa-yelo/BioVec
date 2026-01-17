#!/usr/bin/env python
"""BioVec CLI — embed, index, and search protein sequences."""

import argparse
import sys


def cmd_build_db(args):
    """Build a protein embedding database from FASTA."""
    from biovec.dbbuilder.biovecdb import build_db
    
    db = build_db(
        fasta_file=args.fasta,
        embedder_config=args.embedder_config,
        metric=args.metric,
        save_path=args.output,
    )
    print(f"[biovec] Built database with {db.ntotal:,} sequences → {args.output}.*")


def cmd_search_db(args):
    """Search a database with query sequences."""
    from biovec.dbbuilder.biovecdb import load_db, search_db
    
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
    from biovec.embedders.protein_embedder import embed_protein
    
    out = embed_protein(
        embedder_config=args.embedder_config,
        fasta_file=args.fasta,
        save_h5=args.output,
        verbose=True,
    )
    print(f"[biovec] Embedded {len(out['ids']):,} sequences → {args.output}")


def cmd_gene_call(args):
    """Predict genes from nucleotide FASTA."""
    from biovec.genecaller.pyrodigal_call import run_pyrodigal
    
    run_pyrodigal(
        fasta_path=args.fasta,
        output_dir=args.output,
        mode=args.mode,
        threads=args.threads,
    )


def cmd_csv_to_fasta(args):
    """Convert CSV to FASTA format."""
    from biovec.io.fasta import csv_to_fasta
    
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
    # Parse and run
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
