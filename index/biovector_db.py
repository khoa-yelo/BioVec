import faiss, os, json
import numpy as np
from typing import List
from ..embedders.protein_embedder import ProteinEmbedder
from Bio import SeqIO

class BioVectorDB:
    """Protein embedding index built on FAISS for semantic search."""
    def __init__(self, dim: int = 1280):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[str] = []

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add_fasta(self, fasta_file: str, embedder: ProteinEmbedder):
        vecs, ids = [], []
        for rec in SeqIO.parse(fasta_file, "fasta"):
            emb = embedder.embed(str(rec.seq))
            emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            if emb.shape[1] != self.index.d:
                if self.ntotal == 0:
                    self.index = faiss.IndexFlatL2(emb.shape[1])
                else:
                    raise ValueError("Embedding dimension mismatch.")
            vecs.append(emb); ids.append(rec.id)
        if vecs:
            mat = np.vstack(vecs).astype("float32")
            self.index.add(mat)
            self.metadata.extend(ids)

    def search(self, sequence: str, embedder: ProteinEmbedder, top_k: int = 10):
        q = embedder.embed(sequence).astype("float32").reshape(1, -1)
        if q.shape[1] != self.index.d:
            raise ValueError("Query dimension mismatch.")
        D, I = self.index.search(q, top_k)
        out = []
        for r, (i, d) in enumerate(zip(I[0], D[0]), 1):
            if i == -1: continue
            pid = self.metadata[i] if i < len(self.metadata) else str(i)
            out.append({"rank": r, "id": pid, "distance": float(d)})
        return out

    def save(self, base_path: str):
        faiss.write_index(self.index, f"{base_path}.faiss")
        with open(f"{base_path}.meta.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self, base_path: str):
        self.index = faiss.read_index(f"{base_path}.faiss")
        meta = f"{base_path}.meta.json"
        self.metadata = json.load(open(meta)) if os.path.exists(meta) else []
