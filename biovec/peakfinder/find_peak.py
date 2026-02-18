from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

import numpy as np

from ..dbbuilder.biovecdb import BioVecDB, load_db
from ..io.h5 import read_h5
from ..embedders.protein_embedder import embed_protein, EmbedderConfig
from ..io.fasta import read_fasta


class PeakFinder:
    """Detect peaks and return filtered intervals."""

    def __init__(self, smooth_size=20, prominence=0.1, distance=5, rel_height=0.15, flip_score=True, *, verbose: bool = False):
        self.smooth_size = smooth_size
        self.prominence = prominence
        self.distance = distance
        self.rel_height = rel_height
        self.flip_score = flip_score
        self.verbose = verbose
        if self.verbose:
            print(f"[PeakFinder] initialized: smooth_size={smooth_size}, prominence={prominence}, "
                  f"distance={distance}, rel_height={rel_height}")

    def find_peak(self, data):
        """Find peaks in the data and return filtered intervals."""
        return self.detect_intervals(data)

    def detect_intervals(self, data):
        """
        Detect intervals in the data.
        Filters out intervals that are completely contained within larger intervals.
        """
        if self.verbose:
            print(f"[PeakFinder] processing data: shape={np.asarray(data).shape}, "
                  f"range=[{np.min(data):.4f}, {np.max(data):.4f}]")
        if self.flip_score:
            data = 1-data
            if self.verbose:
                print("[PeakFinder] flipped score, now looking for valleys")
        # Normalize data
        data_range = data.max() - data.min()
        if data_range == 0:
            if self.verbose:
                print("[PeakFinder] data has zero range, returning empty intervals")
            return []
        normalized = (data - data.min()) / data_range
        
        # Smooth data
        smoothed = gaussian_filter1d(normalized, sigma=self.smooth_size)
        if self.verbose:
            print(f"[PeakFinder] normalized and smoothed data")
        
        # Find peaks
        peaks, _props = find_peaks(
            smoothed, prominence=self.prominence, distance=self.distance
        )
        if self.verbose:
            print(f"[PeakFinder] found {len(peaks)} raw peaks at indices: {peaks.tolist()}")
        
        if len(peaks) == 0:
            if self.verbose:
                print("[PeakFinder] no peaks found, returning empty intervals")
            return []
        
        # Compute widths
        _widths, _heights, left_ips, right_ips = peak_widths(
            smoothed, peaks, rel_height=self.rel_height
        )
        
        # Create list of intervals
        intervals = list(zip(left_ips, right_ips))
        if self.verbose:
            print(f"[PeakFinder] computed {len(intervals)} interval widths")
        
        # Sort intervals by size (largest first) and start position
        intervals.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        
        # Filter out nested intervals
        filtered_intervals = []
        for interval in intervals:
            # Check if this interval is contained within any already accepted interval
            is_nested = False
            for accepted in filtered_intervals:
                if interval[0] >= accepted[0] and interval[1] <= accepted[1]:
                    is_nested = True
                    break
            if not is_nested:
                filtered_intervals.append(interval)
        
        if self.verbose and len(filtered_intervals) < len(intervals):
            print(f"[PeakFinder] filtered {len(intervals) - len(filtered_intervals)} nested intervals")
        
        # Sort by start position for display
        filtered_intervals.sort(key=lambda x: x[0])
        result = [(int(interval[0]), int(interval[1])) for interval in filtered_intervals]
        
        if self.verbose:
            print(f"[PeakFinder] returning {len(result)} intervals: {result}")
        
        return result


class RemoteScorer:
    """
    Score queries against a BioVecDB and return similarity of the nearest neighbor.

    Supports query input as:
      - FASTA file path
      - embeddings H5 file (with dataset 'matrix')
      - BioVecDB path (base path) or BioVecDB object
    """

    def __init__(self, db_path: str, use_gpu: bool = False, metric: str = "cosine", verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print(f"[RemoteScorer] initializing with db_path={db_path}, use_gpu={use_gpu}, metric={metric}")
        
        self.db = load_db(db_path, metric=metric, use_gpu=use_gpu) if isinstance(db_path, str) else db_path
        self.metric = metric
        self.use_gpu = use_gpu
        
        if self.verbose:
            print(f"[RemoteScorer] loaded database: dim={self.db.dim}, ntotal={self.db.ntotal}")

    def _embed_fasta(self, fasta_file: str) -> tuple:
        """Embed FASTA and return (matrix, ids, descriptions)."""
        if self.verbose:
            print(f"[RemoteScorer] embedding FASTA file: {fasta_file}")
        
        # Use the database embedder_config for consistency
        cfg = self.db.embedder_config
        if isinstance(cfg, EmbedderConfig):
            embedder_cfg = cfg
        else:
            embedder_cfg = cfg if cfg is not None else "esm2"
        
        records = read_fasta(fasta_file)
        if not records:
            if self.verbose:
                print("[RemoteScorer] no sequences found in FASTA")
            return np.zeros((0, self.db.dim), dtype=np.float32), [], []
        
        ids = [rec[0] for rec in records]
        descriptions = [rec[1] for rec in records]
        seqs = [rec[2] for rec in records]
        
        if self.verbose:
            print(f"[RemoteScorer] embedding {len(seqs)} sequences with {embedder_cfg}")
        
        out = embed_protein(embedder_cfg, sequences=seqs, verbose=self.verbose)
        matrix = np.asarray(out["matrix"], dtype=np.float32)
        
        if self.verbose:
            print(f"[RemoteScorer] embedded {matrix.shape[0]} sequences to dim={matrix.shape[1]}")
        
        return matrix, ids, descriptions

    def _load_embeddings(self, h5_path: str) -> tuple:
        """Load embeddings from H5 and return (matrix, ids, descriptions)."""
        if self.verbose:
            print(f"[RemoteScorer] loading embeddings from H5: {h5_path}")
        
        data = read_h5(h5_path)
        mat = data.get("matrix")
        if mat is None:
            raise ValueError(f"'matrix' dataset not found in {h5_path}")
        
        matrix = np.asarray(mat, dtype=np.float32)
        
        # Try to get IDs and descriptions from H5
        ids = data.get("ids", [f"query_{i}" for i in range(matrix.shape[0])])
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
        
        descriptions = data.get("descriptions", [""] * matrix.shape[0])
        if hasattr(descriptions, "tolist"):
            descriptions = descriptions.tolist()
        descriptions = [d.decode() if isinstance(d, bytes) else str(d) for d in descriptions]
        
        if self.verbose:
            print(f"[RemoteScorer] loaded {matrix.shape[0]} embeddings of dim={matrix.shape[1]}")
        
        return matrix, ids, descriptions

    def _load_query_db(self, db_path: str) -> tuple:
        """Load embeddings from a BioVecDB's H5 file directly (faster than reconstructing from FAISS)."""
        if self.verbose:
            print(f"[RemoteScorer] loading query DB embeddings from: {db_path}.h5")
        
        h5_path = f"{db_path}.h5"
        data = read_h5(h5_path)
        mat = data.get("matrix")
        if mat is None:
            raise ValueError(f"'matrix' dataset not found in {h5_path}")
        
        matrix = np.asarray(mat, dtype=np.float32)
        
        # Get IDs and descriptions from H5
        ids = data.get("ids", [f"query_{i}" for i in range(matrix.shape[0])])
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
        
        descriptions = data.get("descriptions", [""] * matrix.shape[0])
        if hasattr(descriptions, "tolist"):
            descriptions = descriptions.tolist()
        descriptions = [d.decode() if isinstance(d, bytes) else str(d) for d in descriptions]
        
        if self.verbose:
            print(f"[RemoteScorer] loaded {matrix.shape[0]} query embeddings of dim={matrix.shape[1]}")
        
        return matrix, ids, descriptions

    def _get_query_data(self, *, fasta: str = None, embeddings_h5: str = None, 
                         query_db: str = None, query_embeddings: np.ndarray = None,
                         query_ids: list = None) -> tuple:
        """
        Load query data and return (matrix, ids, descriptions).
        """
        if sum(x is not None for x in [fasta, embeddings_h5, query_db, query_embeddings]) != 1:
            raise ValueError("Provide exactly one of fasta, embeddings_h5, query_db, query_embeddings.")

        if fasta is not None:
            Q, ids, descriptions = self._embed_fasta(fasta)
        elif embeddings_h5 is not None:
            Q, ids, descriptions = self._load_embeddings(embeddings_h5)
        elif query_db is not None:
            Q, ids, descriptions = self._load_query_db(query_db)
        else:
            Q = np.asarray(query_embeddings, dtype=np.float32)
            ids = query_ids if query_ids else [f"query_{i}" for i in range(Q.shape[0])]
            descriptions = [""] * Q.shape[0]
            if self.verbose:
                print(f"[RemoteScorer] using provided embeddings: shape={Q.shape}")

        return Q, ids, descriptions

    def score(self, *, fasta: str = None, embeddings_h5: str = None, query_db: str = None, 
              query_embeddings: np.ndarray = None, query_ids: list = None) -> np.ndarray:
        """
        Score queries and return an array of nearest-neighbor similarities (or distances if metric=euclidean).
        """
        Q, _, _ = self._get_query_data(
            fasta=fasta, embeddings_h5=embeddings_h5, query_db=query_db,
            query_embeddings=query_embeddings, query_ids=query_ids
        )

        if Q.size == 0:
            if self.verbose:
                print("[RemoteScorer] empty query, returning empty scores")
            return np.array([], dtype=np.float32)

        # Ensure correct dimension
        if Q.shape[1] != self.db.dim:
            raise ValueError(f"Query dimension {Q.shape[1]} does not match DB dim {self.db.dim}")

        if self.verbose:
            print(f"[RemoteScorer] searching {Q.shape[0]} queries against {self.db.ntotal} database vectors")

        # KNN search (top-1)
        D, _I = self.db.indexer.search(Q, k=1)
        scores = D[:, 0]
        
        if self.verbose:
            print(f"[RemoteScorer] scoring complete: score range=[{scores.min():.4f}, {scores.max():.4f}], "
                  f"mean={scores.mean():.4f}")
        
        # For cosine, D is similarity; for euclidean, D is distance
        return scores

    def score_detailed(self, *, fasta: str = None, embeddings_h5: str = None, 
                       query_db: str = None, query_embeddings: np.ndarray = None,
                       query_ids: list = None) -> list:
        """
        Score queries and return detailed results including neighbor information.
        
        Returns:
            List of dicts with keys: query_id, score, neighbor_id, neighbor_description
        """
        Q, q_ids, _ = self._get_query_data(
            fasta=fasta, embeddings_h5=embeddings_h5, query_db=query_db,
            query_embeddings=query_embeddings, query_ids=query_ids
        )

        if Q.size == 0:
            if self.verbose:
                print("[RemoteScorer] empty query, returning empty results")
            return []

        # Ensure correct dimension
        if Q.shape[1] != self.db.dim:
            raise ValueError(f"Query dimension {Q.shape[1]} does not match DB dim {self.db.dim}")

        if self.verbose:
            print(f"[RemoteScorer] searching {Q.shape[0]} queries against {self.db.ntotal} database vectors")

        # KNN search (top-1)
        D, I = self.db.indexer.search(Q, k=1)
        
        # Get neighbor metadata from database
        # BioVecDB stores metadata as list of dicts with 'id' and 'description' keys
        db_metadata = self.db.metadata
        
        results = []
        for i in range(Q.shape[0]):
            neighbor_idx = int(I[i, 0])
            score = float(D[i, 0])
            
            # Get neighbor ID and description from metadata
            if neighbor_idx < len(db_metadata):
                meta = db_metadata[neighbor_idx]
                neighbor_id = meta.get("id", f"idx_{neighbor_idx}")
                neighbor_desc = meta.get("description", "")
            else:
                neighbor_id = f"idx_{neighbor_idx}"
                neighbor_desc = ""
            
            if isinstance(neighbor_id, bytes):
                neighbor_id = neighbor_id.decode()
            if isinstance(neighbor_desc, bytes):
                neighbor_desc = neighbor_desc.decode()
            
            results.append({
                "query_id": q_ids[i],
                "score": score,
                "neighbor_id": neighbor_id,
                "neighbor_description": neighbor_desc,
            })
        
        if self.verbose:
            scores = D[:, 0]
            print(f"[RemoteScorer] scoring complete: score range=[{scores.min():.4f}, {scores.max():.4f}], "
                  f"mean={scores.mean():.4f}")
        
        return results


def remote_score(
    db_path: str,
    output_path: str,
    *,
    fasta: str = None,
    embeddings_h5: str = None,
    query_db: str = None,
    use_gpu: bool = False,
    metric: str = "cosine",
    verbose: bool = True,
) -> int:
    """
    Score query proteins against a BioVecDB and write results to TSV.
    
    Args:
        db_path: Path to target BioVecDB (without extension).
        output_path: Path to output TSV file.
        fasta: Path to query FASTA file.
        embeddings_h5: Path to query embeddings H5 file.
        query_db: Path to query BioVecDB (without extension).
        use_gpu: Use GPU for FAISS operations.
        metric: Distance metric ('cosine' or 'euclidean').
        verbose: Print progress information.
        
    Returns:
        Number of queries scored.
        
    Output TSV columns:
        - query_id: ID of the query protein
        - score: Similarity score to nearest neighbor (cosine) or distance (euclidean)
        - neighbor_id: ID of the nearest neighbor in the database
        - neighbor_description: Description of the nearest neighbor
        
    Example:
        >>> # Score FASTA against database
        >>> remote_score(
        ...     db_path="databases/uniprot",
        ...     output_path="results.tsv",
        ...     fasta="query_proteins.fasta",
        ... )
        100
        
        >>> # Score embeddings against database
        >>> remote_score(
        ...     db_path="databases/uniprot",
        ...     output_path="results.tsv",
        ...     embeddings_h5="query_embeddings.h5",
        ... )
        50
        
        >>> # Score one database against another
        >>> remote_score(
        ...     db_path="databases/uniprot",
        ...     output_path="results.tsv",
        ...     query_db="databases/vogdb",
        ... )
        1000
    """
    import os
    
    # Validate input
    inputs = [fasta, embeddings_h5, query_db]
    if sum(x is not None for x in inputs) != 1:
        raise ValueError("Provide exactly one of: fasta, embeddings_h5, query_db")
    
    if verbose:
        print(f"[remote_score] Target database: {db_path}")
        if fasta:
            print(f"[remote_score] Query input: FASTA {fasta}")
        elif embeddings_h5:
            print(f"[remote_score] Query input: H5 {embeddings_h5}")
        else:
            print(f"[remote_score] Query input: BioVecDB {query_db}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize scorer
    scorer = RemoteScorer(db_path, use_gpu=use_gpu, metric=metric, verbose=verbose)
    
    # Get detailed scores
    results = scorer.score_detailed(
        fasta=fasta,
        embeddings_h5=embeddings_h5,
        query_db=query_db,
    )
    
    # Write TSV output
    if verbose:
        print(f"[remote_score] Writing {len(results)} results to {output_path}")
    
    with open(output_path, "w") as f:
        # Header
        f.write("query_id\tscore\tneighbor_id\tneighbor_description\n")
        # Data rows
        for row in results:
            query_id = row["query_id"]
            score = row["score"]
            neighbor_id = row["neighbor_id"]
            neighbor_desc = row["neighbor_description"]
            # Escape tabs in description
            neighbor_desc = neighbor_desc.replace("\t", " ") if neighbor_desc else ""
            f.write(f"{query_id}\t{score:.6f}\t{neighbor_id}\t{neighbor_desc}\n")
    
    if verbose:
        print(f"[remote_score] Done. Output written to {output_path}")
    
    return len(results)


def find_peak_in_csv(
    csv_path: str,
    score_col: str,
    *,
    output_path: str = None,
    smooth_size: int = 20,
    prominence: float = 0.1,
    distance: int = 5,
    rel_height: float = 0.15,
    flip_score: bool = True,
    verbose: bool = False,
):
    """
    Run peak detection on a score column from a CSV/TSV file.

    Reads the CSV, extracts the score column, runs PeakFinder, and returns
    the detected intervals plus a copy of the DataFrame with an added
    ``detected`` column (1 inside an interval, 0 outside).

    Args:
        csv_path: Path to CSV or TSV file.
        score_col: Column name containing the score to analyse.
        output_path: If given, write the annotated DataFrame here.
        smooth_size: PeakFinder smooth_size parameter.
        prominence: PeakFinder prominence parameter.
        distance: PeakFinder distance parameter.
        rel_height: PeakFinder rel_height parameter.
        flip_score: If True, detect valleys (1 - score) instead of peaks.
        verbose: Print progress information.

    Returns:
        (intervals, df) where:
          - intervals: list of (start_idx, end_idx) tuples (0-based row indices)
          - df: pandas DataFrame with an added ``detected`` column (1/0)

    Example:
        >>> intervals, df = find_peak_in_csv("scores.tsv", "vogdb_score")
        >>> print(intervals)
        [(10, 45), (120, 160)]
        >>> df[df["detected"] == 1].head()
    """
    import pandas as pd

    # Auto-detect delimiter
    sep = "\t" if csv_path.endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    if score_col not in df.columns:
        raise ValueError(
            f"Column '{score_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    scores = df[score_col].values.astype(np.float64)

    # Drop NaN rows for peak detection but keep full DF
    valid_mask = ~np.isnan(scores)
    if not valid_mask.any():
        df["detected"] = 0
        if output_path:
            _write_df(df, output_path)
        return [], df

    pf = PeakFinder(
        smooth_size=smooth_size,
        prominence=prominence,
        distance=distance,
        rel_height=rel_height,
        flip_score=flip_score,
        verbose=verbose,
    )

    # Run on valid subset mapped back to original indices
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    raw_intervals = pf.detect_intervals(valid_scores)

    # Map intervals back to original DataFrame row indices
    intervals = []
    for start, end in raw_intervals:
        orig_start = int(valid_indices[max(0, start)])
        orig_end = int(valid_indices[min(end, len(valid_indices) - 1)])
        intervals.append((orig_start, orig_end))

    # Build detected column
    detected = np.zeros(len(df), dtype=int)
    for start, end in intervals:
        detected[start : end + 1] = 1
    df["detected"] = detected

    if verbose:
        n_det = int(detected.sum())
        print(
            f"[find_peak_in_csv] {len(intervals)} intervals, "
            f"{n_det}/{len(df)} rows detected"
        )

    if output_path:
        _write_df(df, output_path)
        if verbose:
            print(f"[find_peak_in_csv] wrote {output_path}")

    return intervals, df


def _write_df(df, path: str) -> None:
    sep = "\t" if path.endswith(".tsv") else ","
    df.to_csv(path, sep=sep, index=False)
