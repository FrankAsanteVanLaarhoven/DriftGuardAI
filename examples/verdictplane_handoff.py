"""The DriftGuard -> VerdictPlane handoff, end to end.

Reads a sealed PromotionDecisionRecord (verifying hash, version, and the fail-closed
derivation), derives the lightweight executor-facing ActionProposal, and prints it —
the exact JSON a VerdictPlane-style decision system would receive. The proposal
carries no authority of its own: it is deterministically recomputable from the record
it references (``evidence_ref`` pins decision_id + content_hash), so the sealed record
stays the single auditable source of truth.

    uv run python examples/verdictplane_handoff.py artifacts/promotion_decision.json

Exit codes mirror the proposed action: 0 promote_model, 78 require_human_review,
1 block_deployment, 2 invalid record.
"""

from __future__ import annotations

import sys

from driftguard.contract import build_action_proposal, parse_record, proposal_to_json

EXIT_BY_ACTION = {"promote_model": 0, "require_human_review": 78, "block_deployment": 1}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        record = parse_record(open(argv[0]).read())   # hash + version + derivation
    except (OSError, ValueError) as exc:
        print(f"INVALID record: {exc}", file=sys.stderr)
        return 2

    proposal = build_action_proposal(record, record_path=argv[0])
    print(proposal_to_json(proposal))
    return EXIT_BY_ACTION[proposal.action]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
