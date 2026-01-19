from typing import List, Literal
import numpy as np
import parasail

AlignmentMode = Literal["global", "local"]


def sequence_identity(
    seq1: str,
    seq2: str,
    mode: AlignmentMode = "global",
    gap_open: int = 10,
    gap_extend: int = 1,
    matrix=parasail.blosum62,
) -> float:
    """
    Compute percent identity between two sequences using parasail.

    Parameters
    ----------
    seq1, seq2 : str
        Protein sequences
    mode : "global" | "local"
        Alignment mode
    gap_open, gap_extend : int
        Gap penalties
    matrix : parasail matrix
        Substitution matrix

    Returns
    -------
    float
        Percent identity (0–100)
    """
    if not seq1 or not seq2:
        return 0.0

    if mode == "global":
        result = parasail.nw_stats_scan_16(seq1, seq2, gap_open, gap_extend, matrix)
    elif mode == "local":
        result = parasail.sw_stats_scan_16(seq1, seq2, gap_open, gap_extend, matrix)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")

    if result.length == 0:
        return 0.0

    return 100.0 * result.matches / result.length


def pairwise_identity_matrix(
    sequences: List[str],
    mode: AlignmentMode = "global",
    gap_open: int = 10,
    gap_extend: int = 1,
    matrix=parasail.blosum62,
) -> np.ndarray:
    """
    Compute an NxN symmetric pairwise identity matrix.

    Parameters
    ----------
    sequences : list of str
        Protein sequences
    mode : "global" | "local"

    Returns
    -------
    np.ndarray
        NxN identity matrix
    """
    n = len(sequences)
    mat = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        seq_i = sequences[i]
        for j in range(i, n):
            pid = sequence_identity(
                seq_i,
                sequences[j],
                mode=mode,
                gap_open=gap_open,
                gap_extend=gap_extend,
                matrix=matrix,
            )
            mat[i, j] = pid
            mat[j, i] = pid

    return mat
