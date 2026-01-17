import logging
import sys
from pathlib import Path
from typing import Literal, List

import pyrodigal
import pyrodigal_gv
from Bio import SeqIO

Mode = Literal["bacteria", "virus", "metagenome"]


def run_pyrodigal(
    fasta_path: Path,
    output_dir: Path,
    mode: Mode = "metagenome",
    threads: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Predict CDS from an input FASTA using pyrodigal/pyrodigal_gv.

    Args:
        fasta_path: Path to the input FASTA file.
        output_dir: Directory where outputs will be written.
        mode: Prediction mode - "bacteria" (single genome), "virus", or "metagenome".
        threads: Number of worker threads (only used for metagenome/virus modes).
        logger: Logger for debug/info messages.
    
    Outputs:
        {prefix}.gff       - Gene annotations in GFF3 format
        {prefix}_genes.fna - Nucleotide sequences of predicted genes
        {prefix}_proteins.faa - Amino acid sequences of predicted proteins
    """
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = fasta_path.stem
    gff_file = output_dir / f"{prefix}.gff"
    nuc_fasta = output_dir / f"{prefix}_genes.fna"
    aa_fasta = output_dir / f"{prefix}_proteins.faa"

    # Read all records
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        logger.warning("No sequences found in %s", fasta_path)
        return

    logger.info(f"Mode: {mode} | Sequences: {len(records)} | File: {fasta_path}")

    if mode == "bacteria":
        _run_bacteria_mode(records, gff_file, nuc_fasta, aa_fasta, logger)
    elif mode == "virus":
        _run_virus_mode(records, gff_file, nuc_fasta, aa_fasta, logger)
    elif mode == "metagenome":
        _run_metagenome_mode(records, gff_file, nuc_fasta, aa_fasta, logger)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'bacteria', 'virus', or 'metagenome'.")

    logger.info(f"Gene finding complete! Outputs: {output_dir}")


def _run_bacteria_mode(records, gff_file, nuc_fasta, aa_fasta, logger):
    """Single genome mode - trains on sequences first."""
    logger.info("Initializing GeneFinder (single genome mode)")
    orf_finder = pyrodigal.GeneFinder(meta=False)

    # Concatenate sequences for training
    training_seqs = [str(rec.seq) for rec in records]
    logger.info("Training on %d sequences...", len(training_seqs))
    orf_finder.train(*training_seqs)

    _write_results(records, orf_finder, gff_file, nuc_fasta, aa_fasta, logger)


def _run_virus_mode(records, gff_file, nuc_fasta, aa_fasta, logger):
    """Viral gene prediction using pyrodigal_gv."""
    logger.info("Initializing ViralGeneFinder (metagenomic viral mode)")
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True)

    _write_results(records, orf_finder, gff_file, nuc_fasta, aa_fasta, logger)


def _run_metagenome_mode(records, gff_file, nuc_fasta, aa_fasta, logger):
    """Metagenomic mode - uses pre-trained models."""
    logger.info("Initializing GeneFinder (metagenomic mode)")
    orf_finder = pyrodigal.GeneFinder(meta=True)

    _write_results(records, orf_finder, gff_file, nuc_fasta, aa_fasta, logger)


def _write_results(records, orf_finder, gff_file, nuc_fasta, aa_fasta, logger):
    """Find genes and write GFF, nucleotide, and protein outputs."""
    with open(gff_file, "w") as gff_out, \
         open(nuc_fasta, "w") as nuc_out, \
         open(aa_fasta, "w") as aa_out:

        for record in records:
            genes = orf_finder.find_genes(str(record.seq))
            logger.debug("%s: %d genes found", record.id, len(genes))

            genes.write_gff(gff_out, sequence_id=record.id, include_translation_table=True)
            genes.write_genes(nuc_out, sequence_id=record.id)
            genes.write_translations(aa_out, sequence_id=record.id)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Predict genes from FASTA using pyrodigal")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "-m", "--mode",
        choices=["bacteria", "virus", "metagenome"],
        default="metagenome",
        help="Prediction mode (default: metagenome)",
    )
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")

    args = parser.parse_args()

    run_pyrodigal(
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        mode=args.mode,
        threads=args.threads,
    )
