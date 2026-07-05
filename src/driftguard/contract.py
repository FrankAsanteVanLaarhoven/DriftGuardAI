"""The versioned promotion-decision wire contract.

A ``PromotionDecisionRecord`` is the exportable, auditable artifact of one promotion
decision: every signal the gates saw, every gate verdict with its parameters, the
fail-closed outcome, and a tamper-evident content hash. It is the seam between
DriftGuard (which decides *whether a candidate qualifies*) and any external promotion
executor — a CI human gate, a deployment controller, or a system like VerdictPlane
(which decides *whether the production mutation is allowed*). The contract is plain
JSON: no framework types, no vendor coupling.

Design principles (each deliberate, none incidental):

* **Versioned** — ``schema_version`` is semver. Minor bumps are additive only;
  consumers accept any minor within their major, ignore unknown fields, and MUST
  reject an unknown major.
* **Fail-closed** — the outcome is *derived*, never asserted: any required gate that
  failed (or is missing) forces ``block``. A record cannot claim "promote" while
  carrying a failed required gate.
* **Human gate as a first-class outcome** — ``hold_for_human`` is a terminal decision
  of this record, not an annotation; automated promotion is the opt-in, not the
  default (mirrors the repo's CI flow, and the deployment-gate discipline of
  Breck et al. 2017, "The ML Test Score").
* **Multi-signal** — aggregate scores and recovery/retention (the continual-learning
  trade-off: catastrophic forgetting, McCloskey & Cohen 1989; Kirkpatrick et al.
  2017), slice-level deltas, calibration (ECE — Guo et al. 2017; degradation under
  shift — Ovadia et al. 2019), drift-detection signals (domain-discriminator drift —
  Rabanser et al. 2019), plus an open ``extra_signals`` map for policy-specific
  inputs.
* **Auditable** — ``content_hash`` is a SHA-256 over the canonical JSON, so any
  post-hoc edit is detectable; ``evidence`` carries pointers/digests of the metric
  artifacts the decision was computed from (provenance ethos of Model Cards,
  Mitchell et al. 2019).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1.1.0"   # 1.1: additive optional fields (risk_level, reason_summary,
                           # proposed_by) + the PromotionProposal companion view.

DECISION_PROMOTE = "promote"
DECISION_BLOCK = "block"
DECISION_HOLD_FOR_HUMAN = "hold_for_human"

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

# Deterministic decision -> executor-action mapping for PromotionProposal. This is the
# MODEL-GOVERNANCE intake vocabulary only: runtime remediation (rollbacks, restarts,
# scaling) belongs to the separate Sentinel->VerdictPlane ActionProposal contract
# (verdictplane: pilots/wwt/schema/action_proposal.schema.json) — deliberately a
# different schema with a different name, because a promotion decision is not an
# incident remediation.
ACTION_BY_DECISION = {
    DECISION_PROMOTE: "promote_model",
    DECISION_BLOCK: "block_deployment",
    DECISION_HOLD_FOR_HUMAN: "require_human_review",
}


@dataclass(frozen=True)
class GateOutcome:
    """One gate's verdict. ``required`` gates drive the decision; advisory gates are
    the risk report (they inform the human/policy layer without blocking)."""
    name: str
    passed: bool
    required: bool
    reason: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionDecisionRecord:
    schema_version: str
    decision_id: str
    decided_at: str                      # ISO 8601 UTC
    decision: str                        # promote | block | hold_for_human
    candidate: dict[str, Any]            # identity: uri/version/hash of the candidate
    incumbent: dict[str, Any]
    baseline: dict[str, Any]
    gates: list[GateOutcome]
    signals: dict[str, Any]              # scores, recovery/retention, slices, ECE, drift
    policy: dict[str, Any]               # required gate names, human_required, floors
    evidence: dict[str, Any]             # artifact pointers / digests
    framework: dict[str, Any]            # producing framework + version
    # --- 1.1 additive convenience fields. The GATES remain authoritative: these are
    # --- derived summaries for quick triage/logging, never a substitute for rule 1.
    risk_level: str | None = None        # derived by derive_risk_level (low|medium|high)
    reason_summary: str | None = None    # one-line human-readable summary
    proposed_by: str = "driftguard"
    content_hash: str = ""               # sha256 over canonical JSON (hash field empty)


def derive_decision(gates: list[GateOutcome], human_required: bool = True) -> str:
    """Fail-closed outcome derivation. Any failed required gate — or no required gate
    at all — blocks. Passing candidates go to the human gate unless a policy has
    explicitly opted into automated promotion."""
    required = [g for g in gates if g.required]
    if not required or any(not g.passed for g in required):
        return DECISION_BLOCK
    return DECISION_HOLD_FOR_HUMAN if human_required else DECISION_PROMOTE


def derive_risk_level(gates: list[GateOutcome]) -> str:
    """Deterministic risk summary from gate outcomes (the gates stay authoritative).

    Any failed required gate ⇒ ``high`` (shipping it would be a fail-closed override).
    Otherwise the advisory failures grade the *accepted residual risk* of a pass:
    none ⇒ ``low``; one ⇒ ``medium``; two or more ⇒ ``high``.
    """
    if any(g.required and not g.passed for g in gates) or not any(g.required for g in gates):
        return RISK_HIGH
    advisory_failures = sum(1 for g in gates if not g.required and not g.passed)
    if advisory_failures == 0:
        return RISK_LOW
    return RISK_MEDIUM if advisory_failures == 1 else RISK_HIGH


def summarize_reason(gates: list[GateOutcome], decision: str) -> str:
    """One-line, log-friendly summary; the full reasons stay on the gates."""
    required = [g for g in gates if g.required]
    failed_required = [g.name for g in required if not g.passed]
    advisory_failed = [g.name for g in gates if not g.required and not g.passed]
    if failed_required:
        return f"{decision}: required gate(s) failed: {', '.join(failed_required)}"
    head = f"{decision}: {len(required)}/{len(required)} required gate(s) passed"
    if advisory_failed:
        return f"{head}; advisory failures: {', '.join(advisory_failed)}"
    return f"{head}; no advisory failures"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hashed(payload: dict[str, Any]) -> str:
    unhashed = dict(payload, content_hash="")
    return hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def build_record(*, candidate: dict[str, Any], incumbent: dict[str, Any],
                 baseline: dict[str, Any], gates: list[GateOutcome],
                 signals: dict[str, Any], policy: dict[str, Any] | None = None,
                 evidence: dict[str, Any] | None = None,
                 human_required: bool = True) -> PromotionDecisionRecord:
    """Assemble, derive the fail-closed decision, and seal the content hash."""
    try:
        from importlib.metadata import version
        fw_version = version("driftguard")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        fw_version = "unknown"
    policy = dict(policy or {})
    policy.setdefault("required_gates", [g.name for g in gates if g.required])
    policy.setdefault("human_required", human_required)
    decision = derive_decision(gates, human_required)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": str(uuid.uuid4()),
        "decided_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": decision,
        "candidate": candidate,
        "incumbent": incumbent,
        "baseline": baseline,
        "gates": [asdict(g) for g in gates],
        "signals": signals,
        "policy": policy,
        "evidence": evidence or {},
        "framework": {"name": "driftguard", "version": fw_version},
        "risk_level": derive_risk_level(gates),
        "reason_summary": summarize_reason(gates, decision),
        "proposed_by": "driftguard",
        "content_hash": "",
    }
    payload["content_hash"] = _hashed(payload)
    return PromotionDecisionRecord(
        **{**payload, "gates": [GateOutcome(**g) for g in payload["gates"]]})


def to_json(record: PromotionDecisionRecord, indent: int | None = 2) -> str:
    return json.dumps(asdict(record), indent=indent, ensure_ascii=False)


def parse_record(raw: str | dict[str, Any]) -> PromotionDecisionRecord:
    """Parse + validate a record from the wire.

    Enforces the versioning contract (reject unknown MAJOR, accept any minor/patch,
    ignore unknown fields for forward compatibility), re-derives the decision from the
    carried gates (a record may never claim more than its gates support), and verifies
    the content hash when present.
    """
    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    version = str(payload.get("schema_version", ""))
    major = version.split(".", 1)[0]
    if major != SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(f"unsupported PromotionDecisionRecord schema major "
                         f"'{version}' (supported: {SCHEMA_VERSION})")

    known = {f for f in PromotionDecisionRecord.__dataclass_fields__}
    body = {k: v for k, v in payload.items() if k in known}
    gate_fields = set(GateOutcome.__dataclass_fields__)
    body["gates"] = [GateOutcome(**{k: v for k, v in g.items() if k in gate_fields})
                     for g in body.get("gates", [])]
    record = PromotionDecisionRecord(**body)

    if record.content_hash and _hashed({**payload, "gates": payload.get("gates", [])}) \
            != record.content_hash:
        raise ValueError("PromotionDecisionRecord content hash mismatch — "
                         "record was modified after sealing")
    derived = derive_decision(record.gates,
                              bool(record.policy.get("human_required", True)))
    if record.decision != derived:
        raise ValueError(f"decision '{record.decision}' inconsistent with gates "
                         f"(fail-closed derivation gives '{derived}')")
    return record


# --------------------------------------------------------------------------- #
# PromotionProposal: the lightweight executor-facing view of a decision record.
# A consumer such as VerdictPlane can act on this alone; the referenced record
# (decision_id + content_hash) remains the auditable authority.
# --------------------------------------------------------------------------- #

PROPOSAL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class PromotionProposal:
    schema_version: str
    proposal_id: str
    created_at: str                       # ISO 8601 UTC
    source: str                           # "driftguard" (runtime monitors may differ)
    action: str                           # promote_model | block_deployment | require_human_review
    target: dict[str, Any]                # what to act on (model identity/alias)
    risk_level: str
    reason: str
    evidence_ref: dict[str, Any]          # decision_id + content_hash (+ optional path)
    requires_human: bool


def build_promotion_proposal(record: PromotionDecisionRecord,
                          target: dict[str, Any] | None = None,
                          record_path: str | None = None) -> PromotionProposal:
    """Derive the executor-facing proposal from a (verified) decision record.

    Everything is mapped deterministically — the proposal can always be recomputed
    from the record, so it carries no authority of its own; ``evidence_ref`` pins the
    sealed record it came from.
    """
    evidence_ref: dict[str, Any] = {"decision_id": record.decision_id,
                                    "content_hash": record.content_hash}
    if record_path:
        evidence_ref["path"] = record_path
    return PromotionProposal(
        schema_version=PROPOSAL_SCHEMA_VERSION,
        proposal_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=record.proposed_by or "driftguard",
        action=ACTION_BY_DECISION[record.decision],
        target=target if target is not None else dict(record.candidate),
        risk_level=record.risk_level or derive_risk_level(record.gates),
        reason=record.reason_summary or summarize_reason(record.gates, record.decision),
        evidence_ref=evidence_ref,
        requires_human=record.decision != DECISION_PROMOTE,
    )


def proposal_to_json(proposal: PromotionProposal, indent: int | None = 2) -> str:
    return json.dumps(asdict(proposal), indent=indent, ensure_ascii=False)
