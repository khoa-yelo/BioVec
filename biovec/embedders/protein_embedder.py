from __future__ import annotations

import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

import yaml
from tqdm import tqdm
from Bio import SeqIO

from ..io.h5 import write_h5


ModelType = Literal["esm2", "t5", "glm2"]
LayerSpec = Union[Literal["last", "middle"], int]


@dataclass
class EmbedderConfig:
    model_type: ModelType
    hub_id: str
    device: Optional[str] = None           # "cuda", "cpu", "cuda:0", ...
    dtype: Optional[torch.dtype] = None    # torch.float16 / torch.bfloat16 / None
    batch_size: int = 8
    max_length: int = 2048
    layer: LayerSpec = "last"              # used for esm2 + t5; glm2 uses pooler_output
    trust_remote_code: bool = True

    @staticmethod
    def _parse_dtype(value: Optional[str]) -> Optional[torch.dtype]:
        if value is None:
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported dtype string: {value}")
        return mapping[value]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EmbedderConfig":
        if yaml is None:
            raise ImportError("pyyaml is required to load EmbedderConfig from YAML.")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML must define a mapping of config fields.")
        if "dtype" in data and isinstance(data["dtype"], str):
            data["dtype"] = cls._parse_dtype(data["dtype"])
        return cls(**data)

    @classmethod
    def from_template(cls, name: str) -> "EmbedderConfig":
        template_path = Path(__file__).parent / "templates" / f"{name}.yaml"
        return cls.from_yaml(template_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a JSON/YAML-serializable dict."""
        from dataclasses import asdict
        d = asdict(self)
        # Convert torch.dtype to string
        if d.get("dtype") is not None:
            d["dtype"] = str(d["dtype"]).replace("torch.", "")
        return d

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class ProteinEmbedder:
    """
    Simple unified wrapper for 3 model types:
      - esm2: HF AutoModel + AutoTokenizer, pooled from hidden states (mean, masked)
      - t5:  T5EncoderModel + T5Tokenizer, pooled from chosen hidden layer (mean, masked)
      - glm2: tattabio gLM2 embed model, uses pooler_output; sequences are prefixed with "<+>"
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if cfg.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(cfg.hub_id, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(cfg.hub_id, dtype=cfg.dtype)
        else:
            # esm2 + glm2 both work with Auto*
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.hub_id, trust_remote_code=cfg.trust_remote_code)
            self.model = AutoModel.from_pretrained(cfg.hub_id, trust_remote_code=cfg.trust_remote_code, torch_dtype=cfg.dtype)

        self.model.to(self.device).eval()

    def embed(
        self,
        sequences: Union[str, Sequence[str]],
        *,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        verbose: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        single = isinstance(sequences, str)
        seqs: List[str] = [sequences] if single else list(sequences)

        bs = batch_size or self.cfg.batch_size
        outs: List[torch.Tensor] = []

        if verbose:
            model_name = self.cfg.hub_id
            model_type = self.cfg.model_type
            device = str(self.device)
            print(f"[ProteinEmbedder] model={model_type} hub_id={model_name}")
            print(f"[ProteinEmbedder] device={device} batch_size={bs} sequences={len(seqs)}")

        start = time.time()
        iterator = range(0, len(seqs), bs)
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch = seqs[i : i + bs]
                outs.append(self._embed_batch(batch).cpu())

        embs = torch.cat(outs, dim=0)  # (N, H)
        if single:
            embs = embs[0]
        if return_numpy:
            # numpy doesn't support bfloat16 -> cast first
            if embs.dtype == torch.bfloat16:
                embs = embs.float()
            out = embs.numpy()
        else:
            out = embs

        if verbose:
            elapsed = time.time() - start
            total = len(seqs)
            per_s = total / elapsed if elapsed > 0 else 0.0
            print(f"[ProteinEmbedder] done in {elapsed:.2f}s ({per_s:.1f} seq/s)")
        return out
    # ---------- internals ----------

    def _embed_batch(self, seqs: List[str]) -> torch.Tensor:
        if self.cfg.model_type == "glm2":
            return self._embed_glm2(seqs)
        if self.cfg.model_type == "t5":
            return self._embed_t5(seqs)
        # esm2
        return self._embed_esm2(seqs)

    def _embed_glm2(self, seqs: List[str]) -> torch.Tensor:
        # gLM2 requires "<+>" prefix
        seqs = [s if s.startswith("<+>") else "<+>" + s for s in seqs]

        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # some models/tokenizers emit token_type_ids; gLM2 forward doesn't take it
        enc.pop("token_type_ids", None)

        out = self.model(**enc)
        if getattr(out, "pooler_output", None) is None:
            raise RuntimeError("Expected pooler_output for glm2 model, but it was not returned.")
        return out.pooler_output  # (B, H)

    def _embed_t5(self, seqs: List[str]) -> torch.Tensor:
        # ProtT5 expects space-separated AAs + rare tokens replaced by X
        seqs = [re.sub(r"[UZOB]", "X", s) for s in seqs]
        seqs = [" ".join(list(s)) for s in seqs]

        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hidden = self._pick_layer(out.hidden_states)  # (B, L, H)
        return self._masked_mean(hidden, enc.get("attention_mask"))

    def _embed_esm2(self, seqs: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # If you want middle layer, request hidden states. Otherwise just use last_hidden_state.
        if self.cfg.layer == "last":
            out = self.model(**enc, return_dict=True)
            hidden = out.last_hidden_state
        else:
            out = self.model(**enc, output_hidden_states=True, return_dict=True)
            hidden = self._pick_layer(out.hidden_states)

        return self._masked_mean(hidden, enc.get("attention_mask"))

    def _pick_layer(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # hidden_states: (embeddings, layer1, ..., layerN)
        n = len(hidden_states)
        layer = self.cfg.layer
        if layer == "last":
            return hidden_states[-1]
        if layer == "middle":
            return hidden_states[(n - 1) // 2]
        if isinstance(layer, int):
            if layer < 0 or layer >= n:
                raise ValueError(f"layer index {layer} out of range [0, {n-1}]")
            return hidden_states[layer]
        raise ValueError(f"Unsupported layer spec: {layer}")

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden: (B, L, H)
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)                   # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-6)               # (B, 1)
        return summed / denom


def embed_protein(
    embedder_config: Union[str, EmbedderConfig],
    *,
    fasta_file: Optional[str] = None,
    sequences: Optional[Sequence[str]] = None,
    save_h5: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Embed protein sequences from a FASTA file or a list of sequences.

    Args:
        embedder_config: YAML path, template name, or EmbedderConfig object.
        fasta_file: path to FASTA file.
        sequences: list of protein sequences.
        save_h5: optional output H5 path. Stores datasets: matrix, ids, descriptions.
        batch_size: override embedder batch size.
        verbose: print progress and config info.
    Returns:
        Dict with keys: matrix, ids, descriptions
    """
    if (fasta_file is None and sequences is None) or (fasta_file and sequences):
        raise ValueError("Provide exactly one of fasta_file or sequences.")

    if isinstance(embedder_config, EmbedderConfig):
        cfg = embedder_config
    elif isinstance(embedder_config, str):
        if embedder_config.endswith(".yaml") or embedder_config.endswith(".yml"):
            cfg = EmbedderConfig.from_yaml(embedder_config)
        else:
            cfg = EmbedderConfig.from_template(embedder_config)
    else:
        raise ValueError("embedder_config must be a YAML path, template name, or EmbedderConfig.")

    ids: List[str] = []
    descriptions: List[str] = []
    seqs: List[str] = []

    if fasta_file is not None:
        for rec in SeqIO.parse(fasta_file, "fasta"):
            ids.append(rec.id)
            descriptions.append(getattr(rec, "description", "") or "")
            seqs.append(str(rec.seq))
    else:
        seqs = list(sequences or [])
        ids = [f"seq_{i}" for i in range(len(seqs))]
        descriptions = [""] * len(seqs)

    if not seqs:
        return {"matrix": np.zeros((0, 0), dtype=np.float32), "ids": [], "descriptions": []}

    embedder = ProteinEmbedder(cfg)
    matrix = np.asarray(
        embedder.embed(seqs, batch_size=batch_size, return_numpy=True, verbose=verbose),
        dtype=np.float32,
    )

    if save_h5:
        write_h5(
            save_h5,
            {
                "matrix": matrix,
                "ids": np.asarray(ids, dtype=object),
                "descriptions": np.asarray(descriptions, dtype=object),
            },
        )

    return {"matrix": matrix, "ids": ids, "descriptions": descriptions}


if __name__ == "__main__":
    seqs = [
        "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        "GAVLIPFYWSTNQDEHRKCM",
    ]

    # ESM2 example (last layer)
    esm = ProteinEmbedder(
        EmbedderConfig(
            model_type="esm2",
            hub_id="facebook/esm2_t33_650M_UR50D",
            dtype=torch.float16,
            batch_size=8,
            max_length=2048,
            layer="last",
        )
    )
    print("esm2:", esm.embed(seqs).shape)

    # ProtT5 example (explicit float16 + middle layer)
    t5 = ProteinEmbedder(
        EmbedderConfig(
            model_type="t5",
            hub_id="Rostlab/prot_t5_xl_half_uniref50-enc",
            dtype=torch.float16,  # explicitly float16
            batch_size=4,
            max_length=1024,
            layer="middle",
        )
    )
    print("t5:", t5.embed(seqs).shape)

    # gLM2 example (bf16 + pooler_output)
    glm2 = ProteinEmbedder(
        EmbedderConfig(
            model_type="glm2",
            hub_id="tattabio/gLM2_650M_embed",
            dtype=torch.bfloat16,
            batch_size=8,
            max_length=2048,
        )
    )
    print("glm2:", glm2.embed(seqs).shape)
