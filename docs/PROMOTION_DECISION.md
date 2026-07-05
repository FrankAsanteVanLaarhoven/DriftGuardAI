# The PromotionDecisionRecord wire contract (v1.0.0)

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

## Consuming it (e.g. VerdictPlane, CI, a controller)

```python
from driftguard.contract import parse_record

record = parse_record(open("results_promotion_decision.json").read())  # verifies
if record.decision == "hold_for_human":
    advisory_failures = [g for g in record.gates if not g.required and not g.passed]
    # -> surface record.signals + advisory_failures to the approver, then execute
```

A consumer that only speaks JSON needs three checks: schema major == 1, content hash
verifies, decision matches the fail-closed derivation. Everything else is data.
