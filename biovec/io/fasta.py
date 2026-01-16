from typing import Iterable, List, Tuple

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

