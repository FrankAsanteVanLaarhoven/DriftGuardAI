"""Standalone baseline-gate CLI for CI.

Compares the committed candidate metrics against the baseline metrics on the fixed
holdout and exits non-zero (fail-closed) unless the candidate clears
``baseline_macro_f1 + PROMOTION_MARGIN``. This is the merge/deploy gate.

    python -m driftguard.gate        # exit 0 = promotable, exit 1 = regression
"""

from __future__ import annotations

import json
import sys

from driftguard import registry
from driftguard.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.metrics_path.exists() or not settings.baseline_metrics_path.exists():
        print("Missing metrics.json / baseline_metrics.json — run `make train` first.",
              file=sys.stderr)
        return 2

    candidate = json.loads(settings.metrics_path.read_text())
    baseline = json.loads(settings.baseline_metrics_path.read_text())
    gate = registry.baseline_gate(candidate["macro_f1"], baseline["macro_f1"],
                                  settings.promotion_margin)
    print(gate.reason)
    if gate.passed:
        print("BASELINE GATE PASSED — candidate is promotable.")
        return 0
    print("BASELINE GATE FAILED — regression blocked (fail-closed).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
