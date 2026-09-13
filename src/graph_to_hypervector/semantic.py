"""Ollama text embeddings -> fixed Rademacher projection -> bipolar MAP."""
import json
import os
from urllib.request import Request, urlopen
import torch
import torchhd
from .config import DIMENSIONS, SEED


class OllamaProjection:
    def __init__(self):
        self.url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = "nomic-embed-text"
        self.matrix = None
        self.cache = {}
        self.input_dimensions = None

    def __call__(self, phrase):
        if phrase not in self.cache:
            # Use the same prefix for stored and queried concepts: these are
            # symmetric phrase comparisons, not document/query retrieval prompts.
            payload = json.dumps({"model": self.model,
                                  "input": "clustering: " + phrase}).encode()
            request = Request(self.url + "/api/embed", data=payload,
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
            except Exception as error:
                raise RuntimeError("Start Ollama and run: ollama pull nomic-embed-text") from error
            x = torch.tensor(result["embeddings"][0], dtype=torch.float32)
            x = x / torch.linalg.vector_norm(x)
            if self.matrix is None:
                self.input_dimensions = x.numel()
                generator = torch.Generator().manual_seed(SEED)
                # R is d_text x 4096 with independent equiprobable -1/+1 entries.
                self.matrix = torch.randint(0, 2, (x.numel(), DIMENSIONS), generator=generator).float() * 2 - 1
            if x.numel() != self.input_dimensions:
                raise ValueError("Embedding dimension changed; do not mix model versions.")
            projected = x @ self.matrix
            # Assign exact zeros +1, so every output component is bipolar.
            h_semantic = torch.where(projected >= 0, 1.0, -1.0).as_subclass(torchhd.MAPTensor)
            self.cache[phrase] = h_semantic
        return self.cache[phrase]
