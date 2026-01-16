import numpy as np
import torch
import faiss

class FaissIndexer:
    def __init__(self, dim, metric= 'cosine', use_gpu=True, gpu_device=0, verbose=True):
        """
        Initialize the FAISS indexer.
        Args:
            dim (int): Dimension of the embeddings.
            metric (str): 'cosine' or 'euclidean'.
            use_gpu (bool): Whether to use GPU (default: True).
            gpu_device (int): GPU device id (default: 0).
        """
        self.dim = dim
        self.metric = metric
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.verbose = verbose
        if self.verbose:
            print(f"[FaissIndexer] dim={dim} metric={metric} use_gpu={use_gpu} gpu_device={gpu_device}")
        if metric == 'cosine':
            cpu_index = faiss.IndexFlatIP(dim)
            self._normalize = True
        elif metric == 'euclidean':
            cpu_index = faiss.IndexFlatL2(dim)
            self._normalize = False
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

        if use_gpu:
            if self.verbose:
                print("[FaissIndexer] building GPU index")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_device, cpu_index)
        else:
            if self.verbose:
                print("[FaissIndexer] building CPU index")
            self.index = cpu_index

    def add(self, embeddings):
        """
        Add embeddings to the index.
        Args:
            embeddings (np.ndarray): Array of shape (N, D).
        """
        if self.verbose:
            print(f"[FaissIndexer] add embeddings shape={np.asarray(embeddings).shape}")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._normalize:
            faiss.normalize_L2(embeddings)
            if self.verbose:
                print("[FaissIndexer] normalized embeddings for cosine similarity")
        self.index.add(embeddings)
        if self.verbose:
            print(f"[FaissIndexer] index size={self.index.ntotal}")

    def search(self, queries, k=5):
        """
        Search for k nearest neighbors.
        Args:
            queries (np.ndarray): Query vectors of shape (M, D).
            k (int): Number of neighbors to return.
        Returns:
            distances (np.ndarray): Distances or similarities of shape (M, k).
            indices (np.ndarray): Indices of neighbors of shape (M, k).
        """
        if self.verbose:
            print(f"[FaissIndexer] search queries shape={np.asarray(queries).shape} k={k}")
        queries = np.asarray(queries, dtype=np.float32)
        if self._normalize:
            faiss.normalize_L2(queries)
            if self.verbose:
                print("[FaissIndexer] normalized queries for cosine similarity")
        distances, indices = self.index.search(queries, k)
        if self.verbose:
            print("[FaissIndexer] search complete")
        return distances, indices 

    def save(self, path):
        """
        Save the FAISS index to disk.
        Args:
            path (str): Path to save the index file.
        """
        if self.verbose:
            print(f"[FaissIndexer] saving index to {path}")
        # If index is on GPU, move to CPU before saving
        if hasattr(self.index, 'getDevice'):  # GPU index
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)
        if self.verbose:
            print("[FaissIndexer] save complete")

    @classmethod
    def load(cls, path, metric='cosine', use_gpu=True, gpu_device=0, verbose=False):
        """
        Load a FAISS index from disk.
        Args:
            path (str): Path to the saved index file.
            metric (str): 'cosine' or 'euclidean'.
            use_gpu (bool): Whether to use GPU (default: True).
            gpu_device (int): GPU device id (default: 0).
        Returns:
            FaissKNN instance with loaded index.
        """
        if verbose:
            print(f"[FaissIndexer] loading index from {path}")
        index = faiss.read_index(path)
        if use_gpu:
            if verbose:
                print("[FaissIndexer] moving index to GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_device, index)
        dim = index.d
        obj = cls(dim=dim, metric=metric, use_gpu=False, verbose=verbose)  # will overwrite index
        obj.index = index
        if verbose:
            print("[FaissIndexer] load complete")
        return obj

    def __repr__(self):
        return f"FaissIndexer(dim={self.dim}, metric={self.metric}, use_gpu={self.use_gpu}, gpu_device={self.gpu_device})"
    
    def __str__(self):
        return f"FaissIndexer(dim={self.dim}, metric={self.metric}, use_gpu={self.use_gpu}, gpu_device={self.gpu_device})"

if __name__ == "__main__":
    # Test FaissIndexer with random data
    N, D = 100_000, 128
    np.random.seed(42)
    embeddings = np.random.randn(N, D).astype('float32')
    query = embeddings[0].reshape(1, -1)
    k = 10
    use_gpu = torch.cuda.is_available()

    print("--- Cosine Similarity (GPU) ---")
    indexer_cosine = FaissIndexer(dim=D, metric='cosine', use_gpu=use_gpu)
    indexer_cosine.add(embeddings)
    distances, indices = indexer_cosine.search(query, k)
    print("Indices:", indices)
    print("Cosine similarities:", distances)

    print("\n--- Euclidean Distance (GPU) ---")
    indexer_euclidean = FaissIndexer(dim=D, metric='euclidean', use_gpu=use_gpu)
    indexer_euclidean.add(embeddings)
    distances, indices = indexer_euclidean.search(query, k)
    print("Indices:", indices)
    print("Euclidean distances:", distances)