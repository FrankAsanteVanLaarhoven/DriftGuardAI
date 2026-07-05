# VerdictPlane intake guide — consuming a DriftGuard `PromotionProposal`

Everything the VerdictPlane side needs to implement and test its **model-governance
intake**, without reading DriftGuard's codebase or coordinating deploys. Written for
the WWT pilot; scoped so the intake can be built whenever that repo's work settles.

**Context.** The WWT pilot has **two intakes by design**: VerdictPlane's own
`ActionProposal` (adopted from Sentinel — incident-centric, runtime remediation) and
this one, `PromotionProposal` (model promotion governance). They are different schemas
with different names on purpose; route by name. Rationale:
[`PROMOTION_DECISION.md`](PROMOTION_DECISION.md).

## What arrives

A `PromotionProposal` v1.0.0 (committed fixture:
[`examples/sample_promotion_proposal.json`](../examples/sample_promotion_proposal.json),
generated from the committed sealed record
[`artifacts/promotion_decision.json`](../artifacts/promotion_decision.json)):

```jsonc
{
  "schema_version": "1.0.0",
  "proposal_id": "<uuid4>",
  "created_at": "<ISO 8601 UTC>",
  "source": "driftguard",
  "action": "promote_model",          // or block_deployment | require_human_review
  "target": { /* model identity: kind, algo, seed, mlflow_version */ },
  "risk_level": "low",                // low | medium | high — derived, never asserted
  "reason": "promote: 1/1 required gate(s) passed; no advisory failures",
  "evidence_ref": {"decision_id": "…", "content_hash": "…", "path": "…"},
  "requires_human": false
}
```

The proposal has **zero authority**: it is deterministically recomputable from the
sealed `PromotionDecisionRecord` it references. The record is the auditable truth.

## The intake's validation contract (5 steps, all stdlib)

1. **Parse the proposal**; reject unknown `schema_version` major (supported: `1`).
2. **Fetch the referenced record** via `evidence_ref` (`path`, or however records are
   delivered) and run the **three record checks** — schema major == 1, SHA-256 content
   hash verifies, `decision` equals the fail-closed derivation from `gates`. A
   language-agnostic reference implementation is ~50 lines:
   [`examples/consume_decision.py`](../examples/consume_decision.py) (zero driftguard
   imports; port freely).
3. **Cross-check proposal against record** (because the proposal has no authority):
   `action` must equal the fixed mapping from `record.decision`
   (`promote → promote_model`, `block → block_deployment`,
   `hold_for_human → require_human_review`), and `risk_level` must match the record's.
   Any mismatch ⇒ treat as tampered ⇒ reject.
4. **Route by `action`, default-deny**: unmatched or unknown actions go to
   `require_human` — this is already VerdictPlane's own invariant, so an unknown
   future action from a newer DriftGuard degrades safely.
5. **Ledger the decision**: append `proposal_id`, `evidence_ref.decision_id`, and
   `evidence_ref.content_hash` to the provenance ledger. The content hash is the
   permanent tamper-evident link back to the full gate-by-gate evidence — nothing else
   needs copying.

Steps 1–3 are pure JSON + SHA-256: no model client, no network import — compatible
with VerdictPlane's deterministic-enforcement-path invariant.

## Suggested policy semantics per action

| action | suggested VerdictPlane handling |
|---|---|
| `promote_model` | governed mutation: apply policy (environment, risk_class mapping), then execute or gate |
| `require_human_review` | human gate queue, with `reason` + the record's advisory-gate failures as the review payload |
| `block_deployment` | no-op enforcement + ledger entry (the producer already refused; record it) |

`risk_level` maps naturally onto the pilot's `risk_class` policy dimension
(`low/medium/high`; DriftGuard does not emit `critical`). If the WWT policy needs
`environment`/`target_system`/`blast_radius` for these proposals, they belong in
VerdictPlane's policy configuration for the model-governance intake, not in this
schema — a promotion's blast radius is a property of the serving deployment, which
VerdictPlane knows and DriftGuard does not.

## Library integration (VerdictPlane as a library, not a service)

VerdictPlane is an in-process library + CLI; its one choke point is
`verdictplane.interceptor.govern(action, call, *, policy, ledger, gate)` with a tiny
generic `Action` boundary model (`tool/effect/args/agent/context`). Verified against
its source — three design decisions follow:

1. **Adapter, not a native proposal type.** VerdictPlane's enforcement path stays
   generic and DriftGuard-ignorant. DriftGuard ships the adapter:
   `contract.proposal_to_governed_action(proposal)` emits the plain action dict
   `govern()` validates — **zero verdictplane imports on our side, zero driftguard
   imports on theirs**. The pilot glue is a few lines:

   ```python
   from verdictplane.gate import Gate
   from verdictplane.interceptor import govern
   from verdictplane.policy import load_policy
   from verdictplane.provenance import Ledger

   # steps 1–3 of the validation contract first (verify record + cross-check), then:
   action = proposal_to_governed_action(proposal)          # from driftguard.contract
   govern(action, execute_promotion,                        # side effect runs only if allowed
          policy=load_policy("pilots/wwt/policies/model_governance.yaml"),
          ledger=Ledger("artifacts/ledger"), gate=Gate("artifacts/gate"))
   ```

   `govern()` gives the pilot everything the 5-step contract's steps 4–5 asked for,
   natively: default-`require_human` policy, hash-chained ledger records before/after
   the side effect, and `PolicyDenied`/`ApprovalDenied` fail-closed exceptions.

2. **Human approval: use the real Gate + CLI, no simulator.** The `Gate` is a
   file-backed, cross-process approval queue (quorum votes, deny vetoes, timeout ⇒
   denied — fail-safe). A blocked `govern()` in the pilot process resolves the moment
   a reviewer runs the real `verdictplane` CLI (`pending` / `approve` / `deny`) in
   another terminal. That *is* the demo, and it's real.

3. **Policy: pilot-owned, first-match, default-deny.** A starting policy for this
   intake (dotted paths into the adapter's `args`):

   ```yaml
   default: require_human
   rules:
     - match: {tool: promote_model, args.risk_level: low, args.requires_human: false}
       decision: allow
     - match: {tool: promote_model, args.risk_level: high}
       decision: deny
     - match: {tool: block_deployment}
       decision: deny          # the producer already refused; record it as blocked
   ```

   Note the layering: DriftGuard's `requires_human` is an **input** to policy, not a
   decision — the governor may escalate an auto-promote to its human gate (producer
   proposes, governor disposes). Unmatched actions — including any future DriftGuard
   action this policy has never heard of — fall to `require_human` by default.

## Acceptance check ("consumed one proposal" — the merge criterion)

The `feature/wwt-pilot` branch merges to DriftGuard `main` when VerdictPlane's intake:

1. accepts [`examples/sample_promotion_proposal.json`](../examples/sample_promotion_proposal.json)
   + its referenced record, routes it, and writes a ledger entry;
2. **rejects** the same record with any byte modified (hash check), and a proposal
   whose `action` disagrees with the record's derived decision (cross-check);
3. sends an unknown action value to `require_human` (default-deny).

Fresh inputs can be regenerated at any time:
`make train && uv run python examples/verdictplane_handoff.py artifacts/promotion_decision.json`.
