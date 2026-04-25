# InboxWorld Environment Spec

## Problem statement

InboxWorld keeps the Round 1 problem statement, `email triage`, but models it as an interactive training environment instead of a static classifier benchmark.

## Observation space

Each step exposes:

- current email metadata and body
- partial thread history
- user profile and preferences
- pending tasks
- scenario description

## Action space

The acting policy returns:

- `priority`: `low`, `medium`, or `high`
- `action_type`: `reply`, `draft`, `defer`, or `escalate`
- `extracted_deadline_hours`: integer or `None`
- `rationale`: short natural-language trace for debugging and demo narration

## Transition logic

- The environment reveals one email at a time.
- Each action is compared against scenario-grounded expected outcomes.
- Delayed penalties are approximated through stronger penalties for missed high-priority blockers and incorrect escalation behavior.

## Reward model

Immediate components:

- `+2.0` for correct priority
- `+2.5` for correct action
- `+1.0` for correct deadline extraction
- `-1.0` for unnecessary escalation

Risk-sensitive penalties:

- `-2.0` extra penalty for missing a truly high-priority email
- `-2.0` for wrong action choice
- `-1.5` for wrong priority
- `-0.5` for missing an expected deadline

## Metrics to surface in the demo

- cumulative reward
- priority accuracy
- action accuracy
- missed urgent email count
- unnecessary escalation count

## Round 2 theme mapping

- Primary: `Theme 3 - World Modeling`
- Secondary: `Theme 1 - Multi-Agent Interactions`
- Secondary: `Theme 2 - Long-Horizon Planning`

## Immediate next upgrades

1. Add mutable inbox state across multiple turns.
2. Add reply-quality scoring based on simulated user edits.
3. Add schema drift for sender importance and policy changes.
4. Replace heuristic policies with trainable model-backed policies.
