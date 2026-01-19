"""
Utility functions for BioVecDB.
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..embedders.protein_embedder import EmbedderConfig, ProteinEmbedder, embed_protein


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (N, D)."""
    arr = np.asarray(arr, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def embed_sequences(
    seqs: List[str],
    embedder: Optional[ProteinEmbedder],
    embedder_config: Optional[Union[str, EmbedderConfig, Dict]],
    ids: Optional[List[str]] = None,
    descriptions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Embed sequences using embedder or config, return matrix/ids/descriptions."""
    # Use provided ids/descriptions or generate defaults
    final_ids = ids if ids is not None else [f"seq_{i}" for i in range(len(seqs))]
    final_descriptions = descriptions if descriptions is not None else [""] * len(seqs)
    
    if embedder is not None:
        matrix = ensure_2d(embedder.embed(seqs, return_numpy=True))
        return {
            "matrix": matrix,
            "ids": final_ids,
            "descriptions": final_descriptions,
        }
    if embedder_config is not None:
        # Get embeddings from embed_protein, but use our ids/descriptions
        result = embed_protein(embedder_config, sequences=seqs, verbose=True)
        return {
            "matrix": result["matrix"],
            "ids": final_ids,
            "descriptions": final_descriptions,
        }
    raise ValueError("ProteinEmbedder or embedder_config is required.")


def format_hits(indices: np.ndarray, distances: np.ndarray, metadata: List[Dict]) -> List[Dict]:
    """Format search results into hit list."""
    hits = []
    for rank, (idx, dist) in enumerate(zip(indices.tolist(), distances.tolist()), 1):
        if idx == -1:
            continue
        meta = metadata[idx] if idx < len(metadata) else {"id": str(idx), "description": ""}
        hits.append({
            "rank": rank,
            "id": meta.get("id", str(idx)),
            "description": meta.get("description", ""),
            "distance": float(dist),
        })
    return hits


def load_sequences(base_path: str) -> Dict[str, str]:
    """Load ID to sequence mapping from a saved BioVecDB.
    
    Args:
        base_path: Path prefix to saved BioVecDB files (without extension).
        
    Returns:
        Dict mapping protein ID -> sequence string.
    """
    seq_path = f"{base_path}.sequences.json"
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequences file not found: {seq_path}")
    with open(seq_path) as f:
        return json.load(f)


def get_fasta_path(base_path: str) -> str:
    """Get path to the saved FASTA file from a BioVecDB.
    
    Args:
        base_path: Path prefix to saved BioVecDB files (without extension).
        
    Returns:
        Path to the saved FASTA file.
    """
    fasta_path = f"{base_path}.fasta"
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    return fasta_path


def save_sequences(save_path: str, ids: List[str], seqs: List[str], fasta_file: str) -> None:
    """Save sequences and copy original FASTA file.
    
    Args:
        save_path: Output path prefix (without extension).
        ids: List of sequence IDs.
        seqs: List of sequences.
        fasta_file: Path to original FASTA file to copy.
    """
    # Copy original FASTA file
    shutil.copy(fasta_file, f"{save_path}.fasta")
    
    # Save ID to sequence mapping
    seq_map = {pid: seq for pid, seq in zip(ids, seqs)}
    with open(f"{save_path}.sequences.json", "w") as f:
        json.dump(seq_map, f, indent=2)

