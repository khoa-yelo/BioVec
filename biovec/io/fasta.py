import csv
from typing import Iterable, List, Optional, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_fasta(path: str) -> List[Tuple[str, str, str]]:
    """
    Read a FASTA file.

    Returns:
        List of tuples: (id, description, sequence)
    """
    records: List[Tuple[str, str, str]] = []
    for rec in SeqIO.parse(path, "fasta"):
        records.append((rec.id, rec.description or "", str(rec.seq)))
    return records


def write_fasta(path: str, records: Iterable[Tuple[str, str, str]]) -> None:
    """
    Write a FASTA file.

    Args:
        records: iterable of (id, description, sequence)
    """
    seq_records: List[SeqRecord] = []
    for rid, desc, seq in records:
        seq_records.append(SeqRecord(Seq(seq), id=rid, description=desc or ""))
    SeqIO.write(seq_records, path, "fasta")


def csv_to_fasta(
    csv_path: str,
    fasta_path: str,
    id_col: str,
    seq_col: str,
    desc_col: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> int:
    """
    Convert a CSV file to FASTA format.

    Args:
        csv_path: Path to input CSV file.
        fasta_path: Path to output FASTA file.
        id_col: Column name (if has_header) or index (as string, e.g. "0") for sequence ID.
        seq_col: Column name or index for sequence.
        desc_col: Column name or index for description (optional).
        delimiter: CSV delimiter (default: ",").
        has_header: Whether CSV has a header row (default: True).

    Returns:
        Number of sequences written.
    """
    records: List[Tuple[str, str, str]] = []

    with open(csv_path, "r", newline="") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                rid = row[id_col]
                seq = row[seq_col]
                desc = row.get(desc_col, "") if desc_col else ""
                records.append((rid, desc, seq))
        else:
            reader = csv.reader(f, delimiter=delimiter)
            id_idx = int(id_col)
            seq_idx = int(seq_col)
            desc_idx = int(desc_col) if desc_col else None
            for row in reader:
                rid = row[id_idx]
                seq = row[seq_idx]
                desc = row[desc_idx] if desc_idx is not None else ""
                records.append((rid, desc, seq))

    write_fasta(fasta_path, records)
    return len(records)


def fasta_to_csv(
    fasta_path: str,
    csv_path: str,
    id_col: str = "id",
    seq_col: str = "sequence",
    desc_col: Optional[str] = "description",
    delimiter: str = ",",
    include_length: bool = False,
) -> int:
    """
    Convert a FASTA file to CSV format.

    Args:
        fasta_path: Path to input FASTA file.
        csv_path: Path to output CSV file.
        id_col: Column name for sequence ID (default: "id").
        seq_col: Column name for sequence (default: "sequence").
        desc_col: Column name for description (default: "description").
                  Set to None to exclude description column.
        delimiter: CSV delimiter (default: ",").
        include_length: Include a "length" column with sequence length (default: False).

    Returns:
        Number of sequences written.

    Example:
        >>> fasta_to_csv("proteins.fasta", "proteins.csv")
        100
        >>> fasta_to_csv("proteins.fasta", "proteins.tsv", delimiter="\\t", include_length=True)
        100
    """
    records = read_fasta(fasta_path)

    # Build header
    fieldnames = [id_col]
    if desc_col:
        fieldnames.append(desc_col)
    fieldnames.append(seq_col)
    if include_length:
        fieldnames.append("length")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        for rid, desc, seq in records:
            row = {id_col: rid, seq_col: seq}
            if desc_col:
                row[desc_col] = desc
            if include_length:
                row["length"] = len(seq)
            writer.writerow(row)

    return len(records)

