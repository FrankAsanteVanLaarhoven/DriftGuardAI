"""The PromotionDecisionRecord wire contract: fail-closed derivation, versioning
policy, tamper evidence, and round-trip fidelity."""

import json

import pytest

from driftguard import contract
from driftguard.contract import GateOutcome, build_record, derive_decision, parse_record


def _gates(dual_passed=True):
    return [
        GateOutcome("dual_drift_aware", dual_passed, True, "the deciding gate"),
        GateOutcome("slice_fixed", False, False, "advisory risk report"),
    ]


def _record(**kwargs):
    defaults = dict(
        candidate={"kind": "candidate", "version": "c1"},
        incumbent={"kind": "incumbent", "version": "i1"},
        baseline={"kind": "baseline", "version": "b1"},
        gates=_gates(),
        signals={"retention_ratio": 0.926},
    )
    defaults.update(kwargs)
    return build_record(**defaults)


def test_decision_is_derived_fail_closed():
    # Required gate fails -> block, regardless of anything else.
    assert derive_decision(_gates(dual_passed=False)) == contract.DECISION_BLOCK
    # No required gate at all -> block (a record cannot promote on advisory gates).
    assert derive_decision([GateOutcome("advisory", True, False, "")]) == \
        contract.DECISION_BLOCK
    # Passing + human_required (the default) -> hold, never silent auto-promote.
    assert derive_decision(_gates()) == contract.DECISION_HOLD_FOR_HUMAN
    # Automated promotion is an explicit policy opt-in.
    assert derive_decision(_gates(), human_required=False) == contract.DECISION_PROMOTE


def test_round_trip_preserves_decision_and_gates():
    record = _record()
    parsed = parse_record(contract.to_json(record))
    assert parsed == record
    assert parsed.decision == contract.DECISION_HOLD_FOR_HUMAN
    assert [g.name for g in parsed.gates] == ["dual_drift_aware", "slice_fixed"]
    assert parsed.schema_version == contract.SCHEMA_VERSION
    assert parsed.content_hash  # sealed


def test_tampering_is_detected():
    payload = json.loads(contract.to_json(_record()))

    # Mutating a signal after sealing breaks the content hash.
    tampered = dict(payload, signals={"retention_ratio": 0.999})
    with pytest.raises(ValueError, match="hash mismatch"):
        parse_record(tampered)

    # Re-sealing with an upgraded decision is caught by fail-closed re-derivation.
    lying = json.loads(contract.to_json(_record(gates=_gates(dual_passed=False))))
    lying["decision"] = contract.DECISION_PROMOTE
    lying["content_hash"] = contract._hashed(lying)
    with pytest.raises(ValueError, match="inconsistent with gates"):
        parse_record(lying)


def test_version_policy_minor_ok_major_rejected():
    payload = json.loads(contract.to_json(_record()))

    # A future minor within our major parses (unknown fields ignored) once the seal
    # is recomputed by its producer.
    minor = dict(payload, schema_version="1.9.0", future_field={"x": 1})
    minor["content_hash"] = contract._hashed(minor)
    parsed = parse_record(minor)
    assert parsed.schema_version == "1.9.0"

    # An unknown major is rejected outright.
    major = dict(payload, schema_version="2.0.0")
    major["content_hash"] = contract._hashed(major)
    with pytest.raises(ValueError, match="schema major"):
        parse_record(major)


def test_policy_defaults_are_recorded():
    record = _record()
    assert record.policy["required_gates"] == ["dual_drift_aware"]
    assert record.policy["human_required"] is True
    assert record.framework["name"] == "driftguard"


def _consumer():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import consume_decision
    return consume_decision


def test_stdlib_reference_consumer_agrees_with_the_library():
    consumer = _consumer()
    payload = json.loads(contract.to_json(_record()))
    # The zero-dependency consumer reaches the same verdict as parse_record...
    assert consumer.validate(payload) == contract.DECISION_HOLD_FOR_HUMAN
    # ...and rejects the same tampering.
    with pytest.raises(ValueError, match="hash mismatch"):
        consumer.validate(dict(payload, signals={"retention_ratio": 1.0}))


def test_training_pipeline_emission_end_to_end(tmp_path):
    """emit_decision_record seals a record the stdlib consumer accepts, and the
    decision mirrors the pipeline's real auto_promote behaviour."""
    from driftguard import registry
    from driftguard.config import Settings
    from driftguard.train import emit_decision_record

    # Repeated vocabulary so the TF-IDF min_df pruning keeps terms on a tiny corpus.
    texts = (["world news of global politics today"] * 8
             + ["sports match report from the stadium"] * 8
             + ["business earnings and market story"] * 8
             + ["science and tech research update"] * 8)
    labels = [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8
    pipe = registry.build_baseline_pipeline().fit(texts, labels)
    base_m = {"macro_f1": 0.5, "accuracy": 0.5}
    prim_m = registry.evaluate(pipe, texts, labels)
    gate = registry.baseline_gate(prim_m["macro_f1"], base_m["macro_f1"])

    for auto_promote, expected in ((True, contract.DECISION_PROMOTE),
                                   (False, contract.DECISION_HOLD_FOR_HUMAN)):
        settings = Settings(artifacts_dir=tmp_path, auto_promote=auto_promote)
        decision = emit_decision_record(settings, gate, base_m, prim_m,
                                        incumbent_f1=None, mlflow_info={},
                                        candidate=pipe, xte=texts, yte=labels)
        assert decision == expected
        payload = json.loads((tmp_path / "promotion_decision.json").read_text())
        assert parse_record(payload).decision == expected           # library check
        assert _consumer().validate(payload) == expected            # stdlib check
        assert "candidate_slices_fixed" in payload["signals"]
        assert payload["signals"]["candidate_ece_fixed"] is not None
