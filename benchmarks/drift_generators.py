"""Controlled drift generators for the DriftGuard benchmark.

Each generator takes a labelled base pool and returns a *current window* (list of
strings) with a specific, isolated kind of shift. Everything is seeded so a run is
reproducible. Inspired by the drift-simulation primitives surveyed in Garcia et al.
(2024) for text streams.

Kinds:
    no_drift            in-distribution resample (for false-positive-rate measurement)
    length_truncate     token_count shift only (PSI's home turf)
    class_prior_shift   one topic dominates the window (covariate shift on labels)
    adjective_swap      replace a fraction of tokens with a disjoint "modifier" vocab
    semantic_replace    replace a large fraction of tokens with a disjoint vocabulary
    gradual_topic       inject an increasing fraction of foreign-vocabulary documents
    char_noise          character-level typo/OCR corruption (insert/delete/substitute)
    token_dropout       drop a fraction of tokens (truncated/degraded logging input)
    semantic_rotation   descriptor-PRESERVING semantic shift: frequent in-vocabulary
                        words consistently swapped with other frequent words of the
                        same character length — surface statistics unchanged, meaning
                        scrambled (the case where descriptor-based detectors are
                        structurally blind and only reading the words can help)
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pandas as pd

# A disjoint vocabulary the base news corpus never uses — isolates semantic shift.
FOREIGN = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod tempor "
    "incididunt labore dolore magna aliqua enim minim veniam quis nostrud exercitation"
).split()
MODIFIERS = "azure crimson gilded obsidian velvet spectral molten arcane".split()


def _sample_texts(texts: Sequence[str], n: int, rng: random.Random) -> list[str]:
    return [rng.choice(texts) for _ in range(n)]


def no_drift(pool: pd.DataFrame, n: int, rng: random.Random, severity: float = 0.0) -> list[str]:
    return _sample_texts(pool["text"].tolist(), n, rng)


def length_truncate(pool: pd.DataFrame, n: int, rng: random.Random,
                    severity: float = 0.8) -> list[str]:
    keep = max(1, int(round((1.0 - severity) * 30)))  # higher severity -> shorter
    return [" ".join(t.split()[:keep]) for t in _sample_texts(pool["text"].tolist(), n, rng)]


def class_prior_shift(pool: pd.DataFrame, n: int, rng: random.Random,
                      severity: float = 0.9) -> list[str]:
    # `severity` fraction of the window comes from a single dominant class.
    labels = sorted(pool["label"].unique())
    dominant = labels[rng.randrange(len(labels))]
    dom_texts = pool.loc[pool["label"] == dominant, "text"].tolist()
    other_texts = pool.loc[pool["label"] != dominant, "text"].tolist()
    out = []
    for _ in range(n):
        src = dom_texts if rng.random() < severity else other_texts
        out.append(rng.choice(src))
    return out


def _replace_fraction(text: str, vocab: list[str], frac: float, rng: random.Random) -> str:
    words = text.split()
    for i in range(len(words)):
        if rng.random() < frac:
            words[i] = rng.choice(vocab)
    return " ".join(words) if words else rng.choice(vocab)


def adjective_swap(pool: pd.DataFrame, n: int, rng: random.Random,
                   severity: float = 0.15) -> list[str]:
    return [_replace_fraction(t, MODIFIERS, severity, rng)
            for t in _sample_texts(pool["text"].tolist(), n, rng)]


def semantic_replace(pool: pd.DataFrame, n: int, rng: random.Random,
                     severity: float = 0.6) -> list[str]:
    return [_replace_fraction(t, FOREIGN, severity, rng)
            for t in _sample_texts(pool["text"].tolist(), n, rng)]


def gradual_topic(pool: pd.DataFrame, n: int, rng: random.Random,
                  severity: float = 0.4) -> list[str]:
    # `severity` fraction of docs are entirely foreign-vocabulary (same length dist).
    texts = _sample_texts(pool["text"].tolist(), n, rng)
    out = []
    for t in texts:
        if rng.random() < severity:
            length = len(t.split()) or 1
            out.append(" ".join(rng.choice(FOREIGN) for _ in range(length)))
        else:
            out.append(t)
    return out


def _corrupt_chars(text: str, sev: float, rng: random.Random) -> str:
    out: list[str] = []
    for ch in text:
        if ch != " " and rng.random() < sev:
            r = rng.random()
            if r < 0.34:
                continue                                   # delete
            if r < 0.67:
                out.append(chr(rng.randint(97, 122)))      # substitute
            else:
                out.append(ch)
                out.append(chr(rng.randint(97, 122)))      # insert
        else:
            out.append(ch)
    return "".join(out).strip() or rng.choice(FOREIGN)


def char_noise(pool: pd.DataFrame, n: int, rng: random.Random,
               severity: float = 0.1) -> list[str]:
    # Realistic input corruption (typos/OCR): misspellings become unseen tokens, so the
    # domain classifier catches it while token_count PSI often does not.
    return [_corrupt_chars(t, severity, rng)
            for t in _sample_texts(pool["text"].tolist(), n, rng)]


def token_dropout(pool: pd.DataFrame, n: int, rng: random.Random,
                  severity: float = 0.4) -> list[str]:
    # Degraded/truncated logging: drop a fraction of tokens -> a token_count shift PSI sees.
    out = []
    for t in _sample_texts(pool["text"].tolist(), n, rng):
        words = t.split()
        kept = [w for w in words if rng.random() >= severity]
        out.append(" ".join(kept) if kept else (words[0] if words else rng.choice(FOREIGN)))
    return out


def _rotation_mapping(pool: pd.DataFrame, rng: random.Random,
                      min_count: int = 100) -> dict[str, str]:
    """A consistent word→word rotation within same-length buckets of *frequent*
    in-vocabulary words. Same length keeps token/char/word-length descriptors
    identical; the ``min_count`` floor keeps both source and target words common
    enough that any reasonable reference sample contains them, so oov_rate stays
    put too. Rotating a shuffled bucket by one guarantees a derangement."""
    from collections import Counter

    counts = Counter(w.lower() for t in pool["text"] for w in t.split() if w.isalpha())
    buckets: dict[int, list[str]] = {}
    for w, c in counts.items():
        if c >= min_count:
            buckets.setdefault(len(w), []).append(w)
    mapping: dict[str, str] = {}
    for words in buckets.values():
        if len(words) < 2:
            continue
        shuffled = sorted(words)
        rng.shuffle(shuffled)
        for src, dst in zip(shuffled, shuffled[1:] + shuffled[:1], strict=True):
            mapping[src] = dst
    return mapping


def semantic_rotation(pool: pd.DataFrame, n: int, rng: random.Random,
                      severity: float = 0.5) -> list[str]:
    # Descriptor-preserving semantic drift: every replacement is 1:1, same character
    # length, alphabetic-for-alphabetic, frequent-for-frequent — token_count,
    # char_count, mean_word_len, oov_rate, and non_alpha_rate are all unchanged by
    # construction, yet word frequencies (and meaning) shift consistently. Only a
    # detector that reads the words (the domain classifier) can separate the corpora.
    mapping = _rotation_mapping(pool, rng)
    out = []
    for t in _sample_texts(pool["text"].tolist(), n, rng):
        words = t.split()
        out.append(" ".join(
            mapping[w.lower()] if w.isalpha() and w.lower() in mapping
            and rng.random() < severity else w
            for w in words))
    return out


GENERATORS = {
    "no_drift": no_drift,
    "length_truncate": length_truncate,
    "class_prior_shift": class_prior_shift,
    "adjective_swap": adjective_swap,
    "semantic_replace": semantic_replace,
    "gradual_topic": gradual_topic,
    "char_noise": char_noise,
    "token_dropout": token_dropout,
    "semantic_rotation": semantic_rotation,
}

# Whether each kind is expected to be genuine drift (for scoring detection vs FPR).
IS_DRIFT = {
    "no_drift": False,
    "length_truncate": True,
    "class_prior_shift": True,
    "adjective_swap": True,
    "semantic_replace": True,
    "gradual_topic": True,
    "char_noise": True,
    "token_dropout": True,
    "semantic_rotation": True,
}
