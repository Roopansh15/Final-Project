from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inboxworld import build_pitch_report, default_scenarios, evaluate_policy_set
from inboxworld.policies import baseline_policy, multi_agent_policy


def main() -> None:
    scenarios = default_scenarios()
    evaluations = evaluate_policy_set(
        scenarios,
        {
            "baseline": baseline_policy,
            "multi_agent": multi_agent_policy,
        },
    )
    report = build_pitch_report(scenarios, evaluations)

    output_dir = Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "pitch_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
