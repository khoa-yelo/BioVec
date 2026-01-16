import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..embedders.protein_embedder import EmbedderConfig, ProteinEmbedder, embed_protein
from ..indexer.faiss_indexer import FaissIndexer
from ..io.fasta import read_fasta
from ..io.h5 import write_h5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (N, D)."""
    arr = np.asarray(arr, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _embed_sequences(
    seqs: List[str],
    embedder: Optional[ProteinEmbedder],
    embedder_config: Optional[Union[str, EmbedderConfig, Dict]],
    ids: Optional[List[str]] = None,
    descriptions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Embed sequences using embedder or config, return matrix/ids/descriptions."""
    if embedder is not None:
        matrix = _ensure_2d(embedder.embed(seqs, return_numpy=True))
        return {
            "matrix": matrix,
            "ids": ids or [f"seq_{i}" for i in range(len(seqs))],
            "descriptions": descriptions or [""] * len(seqs),
        }
    if embedder_config is not None:
        return embed_protein(embedder_config, sequences=seqs, verbose=True)
    raise ValueError("ProteinEmbedder or embedder_config is required.")


def _format_hits(indices: np.ndarray, distances: np.ndarray, metadata: List[Dict]) -> List[Dict]:
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


# ---------------------------------------------------------------------------
# BioVecDB Class
# ---------------------------------------------------------------------------

class BioVecDB:
    """Protein embedding index built on FAISS for semantic search."""

    def __init__(
        self,
        dim: int = 512,
        use_gpu: bool = True,
        metric: str = "cosine",
        embedder: Optional[ProteinEmbedder] = None,
        embedder_config: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        self.indexer = FaissIndexer(dim, metric=metric, use_gpu=use_gpu, verbose=True)
        self.metadata: List[Dict[str, str]] = []
        self.embedder = embedder
        self.embedder_config = (
            EmbedderConfig(**embedder_config) if isinstance(embedder_config, dict) else embedder_config
        )

    @property
    def ntotal(self) -> int:
        return self.indexer.index.ntotal

    @property
    def dim(self) -> int:
        return int(self.indexer.index.d)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> None:
        """Add precomputed embeddings and their ids to the index."""
        descriptions = descriptions or [""] * len(ids)
        emb = _ensure_2d(embeddings)

        if emb.shape[0] != len(ids) or len(descriptions) != len(ids):
            raise ValueError("ids/descriptions length must match embeddings rows.")
        if emb.shape[1] != self.dim:
            if self.ntotal == 0:
                self.indexer = FaissIndexer(int(emb.shape[1]), metric=self.indexer.metric, use_gpu=False)
            else:
                raise ValueError("Embedding dimension mismatch.")

        self.indexer.add(emb)
        self.metadata.extend({"id": pid, "description": desc} for pid, desc in zip(ids, descriptions))

    def add_fasta(self, fasta_file: str, embedder: Optional[ProteinEmbedder] = None) -> None:
        """Embed sequences from a FASTA and add them to the index."""
        records = read_fasta(fasta_file)
        if not records:
            return
        ids, descriptions, seqs = map(list, zip(*records))
        out = _embed_sequences(seqs, embedder or self.embedder, self.embedder_config, ids, descriptions)
        self.add_embeddings(out["matrix"], out["ids"], out["descriptions"])

    def search(self, sequence: str, embedder: Optional[ProteinEmbedder] = None, top_k: int = 10) -> List[Dict]:
        """Search index with a single sequence, return ranked hits."""
        out = _embed_sequences([sequence], embedder or self.embedder, self.embedder_config)
        q = _ensure_2d(out["matrix"])
        if q.shape[1] != self.dim:
            raise ValueError("Query dimension mismatch.")
        D, I = self.indexer.search(q, top_k)
        return _format_hits(I[0], D[0], self.metadata)

    def save(self, base_path: str) -> None:
        """Save index and metadata to disk."""
        self.indexer.save(f"{base_path}.faiss")
        with open(f"{base_path}.meta.json", "w") as f:
            json.dump({"metadata": self.metadata, "embedder_config": self.embedder_config}, f)

    @classmethod
    def load(cls, base_path: str, use_gpu: bool = True, metric: str = "cosine") -> "BioVecDB":
        """Load index and metadata from disk."""
        indexer = FaissIndexer.load(f"{base_path}.faiss", metric=metric, use_gpu=use_gpu)
        obj = cls(dim=int(indexer.index.d), metric=metric, use_gpu=use_gpu)
        obj.indexer = indexer

        meta_path = f"{base_path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                obj.metadata = raw.get("metadata", [])
                cfg = raw.get("embedder_config")
                obj.embedder_config = EmbedderConfig(**cfg) if isinstance(cfg, dict) else cfg
            elif raw and isinstance(raw[0], str):
                obj.metadata = [{"id": pid, "description": ""} for pid in raw]
            else:
                obj.metadata = raw or []
        return obj

    def __repr__(self):
        return f"BioVecDB(dim={self.dim}, ntotal={self.ntotal})"


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def build_db(
    fasta_file: str,
    embedder: Optional[ProteinEmbedder] = None,
    *,
    metric: str = "cosine",
    embedder_config: Optional[Union[str, Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
) -> BioVecDB:
    """Build a BioVecDB from a FASTA file (embed + index)."""
    records = read_fasta(fasta_file)
    if not records:
        db = BioVecDB(dim=512, metric=metric, embedder=embedder, embedder_config=embedder_config)
        if save_path:
            db.save(save_path)
        return db

    ids, descriptions, seqs = map(list, zip(*records))
    out = _embed_sequences(seqs, embedder, embedder_config, ids, descriptions)
    matrix = out["matrix"]

    db = BioVecDB(dim=int(matrix.shape[1]), metric=metric, embedder=embedder, embedder_config=embedder_config)
    db.add_embeddings(matrix, out["ids"], out["descriptions"])

    if save_path:
        db.save(save_path)
        write_h5(f"{save_path}.h5", {
            "matrix": matrix,
            "ids": np.asarray(out["ids"], dtype=object),
            "descriptions": np.asarray(out["descriptions"], dtype=object),
        })
    return db


def load_db(base_path: str, *, metric: str = "cosine") -> BioVecDB:
    """Load a previously saved BioVecDB."""
    return BioVecDB.load(base_path, metric=metric)


def search_db(
    db: BioVecDB,
    *,
    query_fasta: Optional[str] = None,
    query_embeddings: Optional[np.ndarray] = None,
    embedder: Optional[ProteinEmbedder] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Search a BioVecDB with FASTA queries or precomputed embeddings."""
    if (query_fasta is None) == (query_embeddings is None):
        raise ValueError("Provide exactly one of query_fasta or query_embeddings.")

    if query_fasta is not None:
        records = read_fasta(query_fasta)
        if not records:
            return []
        q_ids, _, q_seqs = map(list, zip(*records))
        out = _embed_sequences(q_seqs, embedder or db.embedder, db.embedder_config)
        q_embs = _ensure_2d(out["matrix"])
    else:
        q_embs = _ensure_2d(query_embeddings)
        q_ids = [str(i) for i in range(q_embs.shape[0])]

    if q_embs.shape[1] != db.dim:
        raise ValueError("Query dimension mismatch.")

    D, I = db.indexer.search(q_embs, top_k)
    return [{"query_id": qid, "hits": _format_hits(inds, dists, db.metadata)} for qid, inds, dists in zip(q_ids, I, D)]
