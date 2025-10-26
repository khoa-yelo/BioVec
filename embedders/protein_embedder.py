import torch
from transformers import AutoTokenizer, AutoModel

class ProteinEmbedder:
    """Wrapper for pretrained protein language models (ESM, ProtBERT)."""
    def __init__(self, model: str = "esm2_t33_650M"):
        self.model_name = model
        self.hub_id = model if "/" in model else f"facebook/{model}"
        print(f"[BioVector] Loading model: {self.hub_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hub_id)
        self.model = AutoModel.from_pretrained(self.hub_id)
        self.model.eval()

    def embed(self, sequence: str):
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=4096)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
