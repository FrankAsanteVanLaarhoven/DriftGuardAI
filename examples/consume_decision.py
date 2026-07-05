"""Reference consumer for the PromotionDecisionRecord wire contract — stdlib only.

Deliberately imports NOTHING from driftguard: this is the proof that a consumer
(VerdictPlane, a CI job, a deployment controller — in any language) needs only JSON
and this file's three checks to trust a record:

  1. schema major version is one we speak;
  2. the content hash verifies (tamper evidence);
  3. the decision matches the fail-closed derivation from the carried gates
     (a record may never claim more than its gates support).

Exit codes are CI-friendly: 0 = promote, 78 = hold_for_human, 1 = block, 2 = invalid.

    python examples/consume_decision.py artifacts/promotion_decision.json
"""

from __future__ import annotations

import hashlib
import json
import sys

SUPPORTED_MAJOR = "1"


def canonical_hash(payload: dict) -> str:
    unhashed = dict(payload, content_hash="")
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_decision(gates: list[dict], human_required: bool) -> str:
    required = [g for g in gates if g.get("required")]
    if not required or any(not g.get("passed") for g in required):
        return "block"
    return "hold_for_human" if human_required else "promote"


def validate(payload: dict) -> str:
    version = str(payload.get("schema_version", ""))
    if version.split(".", 1)[0] != SUPPORTED_MAJOR:
        raise ValueError(f"unsupported schema major: {version!r}")
    if payload.get("content_hash") and canonical_hash(payload) != payload["content_hash"]:
        raise ValueError("content hash mismatch — record modified after sealing")
    derived = derive_decision(payload.get("gates", []),
                              bool(payload.get("policy", {}).get("human_required", True)))
    if payload.get("decision") != derived:
        raise ValueError(f"decision {payload.get('decision')!r} inconsistent with gates "
                         f"(fail-closed derivation: {derived!r})")
    return derived


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        payload = json.loads(open(argv[0]).read())
        decision = validate(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID record: {exc}", file=sys.stderr)
        return 2

    advisory_failures = [g for g in payload["gates"]
                         if not g.get("required") and not g.get("passed")]
    print(f"decision: {decision}  (schema {payload['schema_version']}, "
          f"id {payload['decision_id'][:8]}, sealed {payload['decided_at']})")
    for g in advisory_failures:
        print(f"  advisory FAIL — {g['name']}: {g['reason']}")
    if decision == "promote":
        return 0
    if decision == "hold_for_human":
        print("  -> route to the approver with the advisory report above")
        return 78  # EX_TEMPFAIL-style: valid, but a human must act
    print("  -> promotion refused (fail-closed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
