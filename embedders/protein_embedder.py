from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer


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
            self.model = T5EncoderModel.from_pretrained(cfg.hub_id, torch_dtype=cfg.dtype)
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
    ) -> Union[np.ndarray, torch.Tensor]:
        single = isinstance(sequences, str)
        seqs: List[str] = [sequences] if single else list(sequences)

        bs = batch_size or self.cfg.batch_size
        outs: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                batch = seqs[i : i + bs]
                outs.append(self._embed_batch(batch).cpu())

        embs = torch.cat(outs, dim=0)  # (N, H)
        if single:
            embs = embs[0]
        if return_numpy:
            # numpy doesn't support bfloat16 -> cast first
            if embs.dtype == torch.bfloat16:
                embs = embs.float()
            return embs.numpy()
        return embs
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
