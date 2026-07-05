# The PromotionDecisionRecord wire contract (v1.1.0)

The exportable, auditable artifact of one promotion decision — the seam between
DriftGuard (*does the candidate qualify?*) and any external promotion executor: a CI
human gate, a deployment controller, or a decision system such as VerdictPlane (*is
the production mutation allowed?*). Producer and consumer share **plain JSON** and this
document; there is no shared library requirement and no vendor coupling.

Implementation: [`src/driftguard/contract.py`](../src/driftguard/contract.py) ·
Emitted example (measured run): [`benchmarks/results_promotion_decision.json`](../benchmarks/results_promotion_decision.json) ·
Tests: [`tests/test_contract.py`](../tests/test_contract.py)

## Shape

```jsonc
{
  "schema_version": "1.0.0",          // semver — see versioning policy
  "decision_id": "<uuid4>",
  "decided_at": "2026-07-05T02:57:31Z",
  "decision": "hold_for_human",       // promote | block | hold_for_human
  "candidate":  { /* identity: kind, uri/version, training provenance */ },
  "incumbent":  { /* identity + headline score */ },
  "baseline":   { /* identity + headline score */ },
  "gates": [                          // every gate verdict, required and advisory
    {"name": "dual_drift_aware", "passed": true,  "required": true,
     "reason": "refreshed 0.9170 >= ...", "params": {"regression_floor": 0.05}},
    {"name": "slice_fixed",      "passed": false, "required": false,
     "reason": "worst slice 'Sci/Tech' delta -0.0849 < -floor", "params": {}}
  ],
  "signals": {                        // everything the gates saw, plus extras
    "macro_f1": {"stale_on_fixed": 0.9197, "candidate_on_fixed": 0.8519, ...},
    "recovery_ratio": 0.968, "retention_ratio": 0.926,
    "slices_fixed": {"World": 0.8597, ...},
    "calibration_ece": {"stale_fixed": 0.0187, "cand_fixed": 0.0697, ...},
    "drift_detection": {"detected": true, "triggered_by": ["domain_classifier", "descriptor_ks"]}
  },
  "policy":   {"required_gates": ["dual_drift_aware"], "human_required": true},
  "evidence": {"results": "benchmarks/results_recovery.json", ...},
  "framework": {"name": "driftguard", "version": "0.1.0"},
  // --- 1.1 additive convenience fields (optional; 1.0 records omit them) ---
  "risk_level": "medium",              // DERIVED from gates — see rule 6
  "reason_summary": "hold_for_human: 1/1 required gate(s) passed; advisory failures: slice_fixed",
  "proposed_by": "driftguard",
  "content_hash": "<sha256 of canonical JSON with this field empty>"
}
```

## The rules that make it a contract

1. **The decision is derived, never asserted.** `decision` must equal the fail-closed
   derivation from `gates`: any failed **required** gate — or no required gate at all —
   ⇒ `block`; all required gates pass ⇒ `hold_for_human` unless the policy explicitly
   set `human_required: false` (automated promotion is the opt-in, not the default).
   `parse_record` re-derives and rejects records that claim more than their gates
   support.
2. **Advisory gates are the risk report.** They never change the decision; they state
   what the decision *accepts* (slice-concentrated forgetting, calibration loss, …) for
   the human or policy layer downstream. In the measured example the dual gate passes
   while `baseline_fixed`, `slice_fixed`, and `calibration_fixed` ride along as FAIL.
3. **Tamper-evident.** `content_hash` = SHA-256 over the canonical JSON (sorted keys,
   compact separators, `content_hash` emptied). Consumers verify before trusting;
   re-sealing with an inconsistent decision is still caught by rule 1.
4. **Versioning (semver).**
   - **Minor/patch bumps are additive only** — new optional fields, never changed
     meaning. Consumers accept any minor within their major and **ignore unknown
     fields** (forward compatible).
   - **Major bumps may break** — consumers MUST reject an unknown major.
5. **Multi-signal, open-ended.** `signals` carries whatever the gates consumed —
   aggregate scores, recovery/retention, slices, ECE, drift attribution — and
   producers may add custom keys freely (rule 4 makes that safe).
6. **Summaries never outrank gates.** The 1.1 convenience fields are *derived*:
   `risk_level` follows a fixed rule (required-gate failure or no required gate ⇒
   `high`; else 0/1/≥2 advisory failures ⇒ `low`/`medium`/`high`), `reason_summary`
   condenses the gate reasons. They exist for triage and logs; a consumer that acts
   on them without honouring rule 1 is out of contract. (There is deliberately **no**
   `action` field on the record — an assertable action that could disagree with the
   derived `decision` would undo rule 1. Actions live on the `PromotionProposal` view.)

## Research grounding

The record encodes, as first-class fields, concepts with established literature behind
them: the recovery/retention trade-off from catastrophic-forgetting research
(McCloskey & Cohen 1989; Kirkpatrick et al. 2017), calibration and its degradation
under distribution shift (Guo et al. 2017; Ovadia et al. 2019), domain-discriminator
drift detection (Rabanser, Günnemann & Lipton 2019), deployment-gate discipline and
staged rollout checks (Breck et al. 2017, *The ML Test Score*), and the
provenance/reporting ethos of Model Cards (Mitchell et al. 2019). The contract's job
is to carry those measurements to the decision boundary intact — not to invent new
metrics at serialization time.

## Producing it

The real pipeline emits records, not just the benchmark: `make train` (and the drift
retrain path, which reuses the same entrypoint) seals every promotion decision to
`artifacts/promotion_decision.json` (`emit_decision_record` in `driftguard.train`;
disable with `DRIFTGUARD_EMIT_DECISION_RECORD=false`). The decision mirrors the
pipeline's real behaviour: with `auto_promote` on, a gate-passing candidate records
`promote`; the drift pipeline runs with it off and records `hold_for_human`. Candidate
slice scores and calibration ride along as signals; the evidence block carries SHA-256
digests of the metric files the gate read.

## Consuming it (e.g. VerdictPlane, CI, a controller)

Python consumers can use the library:

```python
from driftguard.contract import parse_record

record = parse_record(open("artifacts/promotion_decision.json").read())  # verifies
if record.decision == "hold_for_human":
    advisory_failures = [g for g in record.gates if not g.required and not g.passed]
    # -> surface record.signals + advisory_failures to the approver, then execute
```

But no consumer *needs* the library. A consumer that only speaks JSON performs three
checks: schema major == 1, content hash verifies, decision matches the fail-closed
derivation. [`examples/consume_decision.py`](../examples/consume_decision.py) is the
**reference consumer — stdlib only, zero driftguard imports** — proving those checks in
~50 lines, with CI-friendly exit codes (`0` promote · `78` hold_for_human · `1` block ·
`2` invalid record). A test asserts the stdlib consumer and `parse_record` reach
identical verdicts and reject identical tampering.

```bash
uv run python examples/consume_decision.py artifacts/promotion_decision.json
```

## PromotionProposal — the executor-facing view (v1.0.0)

A decision system like VerdictPlane does not need the full record to *route* work; it
needs what to do, to what, at what risk, and where the proof lives.
`build_promotion_proposal(record)` derives exactly that:

```jsonc
{
  "schema_version": "1.0.0",
  "proposal_id": "<uuid4>",
  "created_at": "2026-07-05T13:43:27Z",
  "source": "driftguard",
  "action": "require_human_review",   // promote_model | block_deployment | require_human_review
  "target": { /* the record's candidate identity (or an explicit override) */ },
  "risk_level": "medium",
  "reason": "hold_for_human: 1/1 required gate(s) passed; advisory failures: slice_fixed",
  "evidence_ref": {"decision_id": "…", "content_hash": "…", "path": "artifacts/promotion_decision.json"},
  "requires_human": true
}
```

Properties, by design:

- **Zero authority.** The mapping is deterministic (`decision → action` via a fixed
  table; risk and reason are the record's derived fields), so a proposal is always
  recomputable from its record. `evidence_ref` pins the sealed record; a consumer that
  wants proof follows the reference and runs the three record checks.
- **Two intakes, deliberately.** VerdictPlane's WWT pilot defines its own
  `ActionProposal` (adopted verbatim from Sentinel: incident-centric — `incident`,
  `root_service`, `evidence_grounding`, runtime-remediation actions like
  `rollback_change`/`restart_service`). A model-promotion decision does not fit those
  semantics, so DriftGuard's intake is this **differently-named, domain-specific**
  `PromotionProposal` — no field abuse, no schema collision, and VerdictPlane routes
  the two by name.
- [`examples/verdictplane_handoff.py`](../examples/verdictplane_handoff.py) is the
  end-to-end handoff: verify record → derive proposal → emit JSON, with the same
  CI-friendly exit codes as the reference consumer.
- [`PROMOTION_PROPOSAL_INTAKE.md`](PROMOTION_PROPOSAL_INTAKE.md) is the
  **implementation guide for the consuming side** (VerdictPlane): the 5-step
  validation contract, policy semantics per action, a committed fixture, and the
  acceptance check that clears this branch's merge criterion.

## Independence guarantees

The contract is the *entire* coupling surface between producer and consumer:

- **DriftGuard produces** records; it never calls a consumer, and it does not know or
  care what executes the promotion.
- **A consumer executes** (or refuses) production mutations; it needs only this
  document and JSON — any language, no shared library, no shared deploy cadence.
- Either side can version independently: minors are additive and ignorable; a major
  bump is an explicit, deliberate breaking event both sides must opt into.
