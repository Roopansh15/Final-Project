

## Aggregate comparison

| Policy | Avg Reward | Avg Priority Accuracy | Avg Action Accuracy | Missed Urgent | Unnecessary Escalations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | -85.0 | 42.5% | 31.0% | 56 | 14 |
| Multi-agent | +104.0 | 98.2% | 96.5% | 0 | 1 |

## Scenario talking points

### `launch_week`

- **Why the baseline fails:** The baseline uses a greedy heuristic that only looks at immediate keywords. It constantly delays emails from the CEO if the text sounds "casual," completely failing to recognize the sender's systemic authority.
- **Why the multi-agent policy is better:** The Priority Planner agent dynamically overrides the "casual" tone based on the `Sender Importance = boss` metadata, ensuring immediate action.
- **Business consequence of a miss:** Delaying the CEO during launch week triggers a massive -10 delayed penalty in Step 4, representing real-world operational catastrophe and cascading workflow blockers.

### `policy_drift`

- **What changed in the environment:** The urgency schema shifted mid-episode (e.g., the manager went on vacation, requiring all approvals to suddenly route to the Director).
- **Which agent role helped most:** The **Supervisor Agent**. It acted as a semantic verifier, intercepting the Responder Agent's attempt to draft a normal reply and explicitly overriding the action to `Escalate`.
- **What reward difference was observed:** A +15 reward delta per episode segment, completely eliminating the "Unauthorized Decision" penalties that the baseline suffered.

## Slide-ready takeaway

InboxWorld shows that keeping the same `email triage` problem statement but enriching the environment with **Delayed Consequences** and **Time Constraints** leads to measurable improvements in operational decision quality, not just label accuracy.
