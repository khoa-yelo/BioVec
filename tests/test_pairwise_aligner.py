#!/usr/bin/env python
"""
Test script for the pairwise_aligner module.

Usage:
    python tests/test_pairwise_aligner.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from biovec.seqaligner.pairwise_aligner import sequence_identity, pairwise_identity_matrix


# Sample protein sequences for testing
SEQUENCES = {
    "seq1": "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQSLTQSRTLLTGKSGLVKGN",
    "seq2": "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQSLTQSRTLLTGKSGLVKGN",  # identical
    "seq3": "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQSLTQSRTLLTGKSGLVKXX",  # 2 aa diff
    "seq4": "MKTIIALSYIFCLVFADYKNTGKCKSGTSTFNVHSSAKGTSVASLKGNSN",  # different
    "seq5": "ACDEFGHIKLMNPQRSTVWY",  # all amino acids
    "seq6": "ACDEFGHIKLMNPQRSTVWY",  # identical to seq5
    "seq7": "XXXXXXXXXXXXXXXXXXXX",  # all X (unknown)
}


def test_sequence_identity_identical():
    """Test identity between identical sequences."""
    print("=" * 60)
    print("TEST: Identical sequences")
    print("=" * 60)
    
    # Global alignment
    pid_global = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq2"], mode="global")
    print(f"  Global alignment: {pid_global:.2f}%")
    assert pid_global == 100.0, f"Expected 100%, got {pid_global}%"
    
    # Local alignment
    pid_local = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq2"], mode="local")
    print(f"  Local alignment:  {pid_local:.2f}%")
    assert pid_local == 100.0, f"Expected 100%, got {pid_local}%"
    
    print("  ✓ PASSED")
    print()


def test_sequence_identity_similar():
    """Test identity between similar sequences (2 aa difference)."""
    print("=" * 60)
    print("TEST: Similar sequences (2 aa difference)")
    print("=" * 60)
    
    pid_global = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq3"], mode="global")
    print(f"  Global alignment: {pid_global:.2f}%")
    # Expect ~97% (65/67 or similar)
    assert 95.0 <= pid_global <= 100.0, f"Expected ~97%, got {pid_global}%"
    
    pid_local = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq3"], mode="local")
    print(f"  Local alignment:  {pid_local:.2f}%")
    
    print("  ✓ PASSED")
    print()


def test_sequence_identity_different():
    """Test identity between different sequences."""
    print("=" * 60)
    print("TEST: Different sequences")
    print("=" * 60)
    
    pid_global = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq4"], mode="global")
    print(f"  Global alignment: {pid_global:.2f}%")
    # Expect low identity
    assert pid_global < 50.0, f"Expected <50%, got {pid_global}%"
    
    pid_local = sequence_identity(SEQUENCES["seq1"], SEQUENCES["seq4"], mode="local")
    print(f"  Local alignment:  {pid_local:.2f}%")
    
    print("  ✓ PASSED")
    print()


def test_sequence_identity_empty():
    """Test identity with empty sequences."""
    print("=" * 60)
    print("TEST: Empty sequences")
    print("=" * 60)
    
    pid1 = sequence_identity("", SEQUENCES["seq1"], mode="global")
    print(f"  Empty vs seq1: {pid1:.2f}%")
    assert pid1 == 0.0, f"Expected 0%, got {pid1}%"
    
    pid2 = sequence_identity(SEQUENCES["seq1"], "", mode="global")
    print(f"  seq1 vs empty: {pid2:.2f}%")
    assert pid2 == 0.0, f"Expected 0%, got {pid2}%"
    
    pid3 = sequence_identity("", "", mode="global")
    print(f"  Empty vs empty: {pid3:.2f}%")
    assert pid3 == 0.0, f"Expected 0%, got {pid3}%"
    
    print("  ✓ PASSED")
    print()


def test_sequence_identity_self():
    """Test self-identity."""
    print("=" * 60)
    print("TEST: Self identity")
    print("=" * 60)
    
    for name, seq in SEQUENCES.items():
        if seq:  # skip empty
            pid = sequence_identity(seq, seq, mode="global")
            print(f"  {name}: {pid:.2f}%")
            assert pid == 100.0, f"Expected 100% for {name}, got {pid}%"
    
    print("  ✓ PASSED")
    print()


def test_sequence_identity_modes():
    """Test global vs local alignment modes."""
    print("=" * 60)
    print("TEST: Global vs Local alignment modes")
    print("=" * 60)
    
    # Create sequences where local should find better alignment
    # Global penalizes gaps at ends, local finds best local match
    long_seq = "AAAAAAAAAA" + SEQUENCES["seq5"] + "AAAAAAAAAA"
    short_seq = SEQUENCES["seq5"]
    
    pid_global = sequence_identity(long_seq, short_seq, mode="global")
    pid_local = sequence_identity(long_seq, short_seq, mode="local")
    
    print(f"  Long seq (with flanking A's) vs short seq:")
    print(f"    Global: {pid_global:.2f}%")
    print(f"    Local:  {pid_local:.2f}%")
    
    # Local should find the perfect match region
    assert pid_local >= pid_global, "Local should be >= global for this case"
    
    print("  ✓ PASSED")
    print()


def test_pairwise_identity_matrix_basic():
    """Test pairwise identity matrix computation."""
    print("=" * 60)
    print("TEST: Pairwise identity matrix")
    print("=" * 60)
    
    seqs = [SEQUENCES["seq1"], SEQUENCES["seq2"], SEQUENCES["seq3"], SEQUENCES["seq4"]]
    
    mat = pairwise_identity_matrix(seqs, mode="global")
    
    print(f"  Matrix shape: {mat.shape}")
    print(f"  Matrix dtype: {mat.dtype}")
    print()
    print("  Identity matrix:")
    print("       seq1    seq2    seq3    seq4")
    for i, name in enumerate(["seq1", "seq2", "seq3", "seq4"]):
        row = " ".join(f"{mat[i, j]:6.1f}" for j in range(4))
        print(f"  {name} {row}")
    
    # Check symmetry
    assert np.allclose(mat, mat.T), "Matrix should be symmetric"
    print()
    print("  ✓ Matrix is symmetric")
    
    # Check diagonal is 100%
    assert np.allclose(np.diag(mat), 100.0), "Diagonal should be 100%"
    print("  ✓ Diagonal is 100%")
    
    # Check seq1 == seq2 (identical)
    assert mat[0, 1] == 100.0, "seq1 and seq2 should be 100% identical"
    print("  ✓ Identical sequences have 100% identity")
    
    print("  ✓ PASSED")
    print()


def test_pairwise_identity_matrix_single():
    """Test pairwise matrix with single sequence."""
    print("=" * 60)
    print("TEST: Pairwise matrix with single sequence")
    print("=" * 60)
    
    mat = pairwise_identity_matrix([SEQUENCES["seq1"]], mode="global")
    
    print(f"  Matrix shape: {mat.shape}")
    print(f"  Matrix value: {mat[0, 0]:.2f}%")
    
    assert mat.shape == (1, 1), "Should be 1x1 matrix"
    assert mat[0, 0] == 100.0, "Self-identity should be 100%"
    
    print("  ✓ PASSED")
    print()


def test_pairwise_identity_matrix_local():
    """Test pairwise matrix with local alignment."""
    print("=" * 60)
    print("TEST: Pairwise matrix with local alignment")
    print("=" * 60)
    
    seqs = [SEQUENCES["seq5"], SEQUENCES["seq6"], SEQUENCES["seq7"]]
    
    mat_global = pairwise_identity_matrix(seqs, mode="global")
    mat_local = pairwise_identity_matrix(seqs, mode="local")
    
    print("  Global alignment matrix:")
    for i in range(3):
        row = " ".join(f"{mat_global[i, j]:6.1f}" for j in range(3))
        print(f"    {row}")
    
    print()
    print("  Local alignment matrix:")
    for i in range(3):
        row = " ".join(f"{mat_local[i, j]:6.1f}" for j in range(3))
        print(f"    {row}")
    
    assert np.allclose(mat_global, mat_global.T), "Global matrix should be symmetric"
    assert np.allclose(mat_local, mat_local.T), "Local matrix should be symmetric"
    
    print("  ✓ PASSED")
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PAIRWISE ALIGNER TESTS")
    print("=" * 60)
    print()
    
    test_sequence_identity_identical()
    test_sequence_identity_similar()
    test_sequence_identity_different()
    test_sequence_identity_empty()
    test_sequence_identity_self()
    test_sequence_identity_modes()
    test_pairwise_identity_matrix_basic()
    test_pairwise_identity_matrix_single()
    test_pairwise_identity_matrix_local()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

