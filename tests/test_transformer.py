"""The DistilBERT wrapper must present a scikit-learn-like surface and keep its heavy
imports lazy, so the core install/lint/tests never require torch."""

from driftguard.transformer_primary import TransformerClassifier, build_transformer_pipeline


def test_wrapper_constructs_without_torch_installed():
    clf = TransformerClassifier(epochs=1, seed=7)
    # Constructing must not import torch/transformers (lazy) — this runs in core CI.
    assert clf.model_name == "distilbert-base-uncased"
    assert clf.num_labels == 4
    assert clf.seed == 7
    assert clf._model is None and clf._tokenizer is None


def test_wrapper_exposes_sklearn_like_api():
    clf = TransformerClassifier()
    for method in ("fit", "predict", "predict_proba"):
        assert callable(getattr(clf, method))


def test_factory_uses_label_count_and_seed():
    clf = build_transformer_pipeline()
    assert clf.num_labels == 4  # AG News classes
