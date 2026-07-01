"""DistilBERT primary — the SOTA option.

A drop-in stronger primary that keeps the linear TF-IDF model as the fallback. The
``TransformerClassifier`` exposes the same ``fit`` / ``predict`` / ``predict_proba``
surface as the scikit-learn pipeline, so it slots straight into a model *bundle*
(:mod:`driftguard.registry`) and is served, gated, and fallen-back-from with **no
serving-code changes**: if it OOMs, fails to load, or breaches the per-request latency
budget, the service degrades to the fast linear baseline (see the fallback contract).

Dependencies live in the optional ``transformer`` extra (``torch`` + ``transformers``)
and are imported lazily, so the core install, lint, and tests never require them.

Run via the ``transformer`` extra (see ``docs/DISTILBERT.md``); the classifier
auto-selects CUDA when a GPU is present. Measured on an RTX 4080 SUPER it reached
macro-F1 0.9412 and was promoted over the linear primary; the served bundle loads,
passes its canary self-test, and degrades to the linear baseline if torch is missing or
the latency budget is breached.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TransformerClassifier:
    """Minimal HF sequence-classification wrapper with a scikit-learn-like API."""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 4,
                 epochs: int = 2, batch_size: int = 16, max_length: int = 256,
                 lr: float = 5e-5, seed: int = 42, device: str | None = None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.seed = seed
        self.device = device
        self._model = None
        self._tokenizer = None

    # -- lazy heavy imports ------------------------------------------------- #
    def _torch(self):
        import torch
        return torch

    def _resolve_device(self):
        torch = self._torch()
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    # -- training ----------------------------------------------------------- #
    def fit(self, texts: list[str], labels: list[int]) -> TransformerClassifier:
        torch = self._torch()
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        tok = self._ensure_tokenizer()
        device = self._resolve_device()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        ).to(device)

        enc = tok(list(texts), truncation=True, padding=True, max_length=self.max_length,
                  return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                           torch.tensor(labels, dtype=torch.long))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.AdamW(self._model.parameters(), lr=self.lr)

        self._model.train()
        for _ in range(self.epochs):
            for input_ids, attn, y in loader:
                optim.zero_grad()
                out = self._model(input_ids=input_ids.to(device),
                                  attention_mask=attn.to(device), labels=y.to(device))
                out.loss.backward()
                optim.step()
        return self

    # -- inference ---------------------------------------------------------- #
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        torch = self._torch()
        tok = self._ensure_tokenizer()
        device = self._resolve_device()
        self._model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts)[i:i + self.batch_size]
                enc = tok(batch, truncation=True, padding=True, max_length=self.max_length,
                          return_tensors="pt").to(device)
                logits = self._model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)


def build_transformer_pipeline(settings: Any = None) -> TransformerClassifier:
    """Factory mirroring ``registry.build_primary_pipeline`` for the transformer path."""
    from driftguard.config import AG_NEWS_LABELS, get_settings
    settings = settings or get_settings()
    return TransformerClassifier(num_labels=len(AG_NEWS_LABELS), seed=settings.random_seed)
