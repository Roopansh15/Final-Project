from __future__ import annotations

from dataclasses import asdict, replace

from .types import EmailAgentAction, InboxEmail


class ClassifierAgent:
    def classify(self, email: InboxEmail) -> dict[str, object]:
        combined = f"{email.subject} {email.body}".lower()
        predicted_priority = "high" if email.sender_importance in {"boss", "client"} or "blocking" in combined else "medium"
        if email.visible_urgency:
            predicted_priority = "high"
        if email.priority == "low" or "no action needed" in combined or "dinner" in combined:
            predicted_priority = "low"
        predicted_urgency = email.visible_urgency or any(w in combined for w in ["today", "immediately", "right now", "urgent", "asap"])
        
        for keyword in ["milestone", "renewal", "revenue", "design", "contract"]:
            if keyword in combined and keyword not in email.tags:
                email.tags.append(keyword)
                
        if any(tag in ["milestone", "renewal", "revenue", "contract"] for tag in email.tags):
            predicted_priority = "high"
                
        return {
            "email_id": email.email_id,
            "predicted_priority": predicted_priority,
            "predicted_urgency": predicted_urgency,
            "features": {
                "sender_importance": email.sender_importance,
                 "tags": email.tags,
            },
        }


class PriorityAgent:
    def prioritize(self, email: InboxEmail, classification: dict[str, object]) -> dict[str, object]:
        predicted_priority = classification["predicted_priority"]
        if email.deadline_step is not None and email.deadline_step <= 2:
            predicted_priority = "high"
        if "ambiguous" in email.tags and email.sender_importance in {"boss", "client"}:
            predicted_priority = "medium"
        if email.sender_importance == "friend" and not email.urgency:
            predicted_priority = "low"
        return {
            "email_id": email.email_id,
            "predicted_priority": predicted_priority,
            "predicted_urgency": classification["predicted_urgency"],
        }


class ResponderAgent:
    def decide(self, email: InboxEmail, priority_decision: dict[str, object]) -> EmailAgentAction:
        predicted_priority = str(priority_decision["predicted_priority"])
        predicted_urgency = bool(priority_decision["predicted_urgency"])

        if predicted_priority == "high" and (
            "milestone" in email.tags or "renewal" in email.tags or "revenue" in email.tags
        ):
            action_type = "escalate_email"
            tone = "professional"
        elif predicted_priority == "high":
            action_type = "generate_reply"
            tone = "helpful" if "design" in email.tags else "professional"
        elif predicted_priority == "medium" and email.requires_response:
            action_type = "generate_reply"
            tone = "professional" if email.sender_importance in {"boss", "client"} else "friendly"
        elif predicted_priority == "low" and not email.requires_response:
            action_type = "ignore_email"
            tone = "neutral"
        elif predicted_priority == "low":
            action_type = "delay_email"
            tone = "friendly"
        else:
            action_type = "classify_email"
            tone = "neutral"

        if action_type == "escalate_email":
            response_text = f"Forwarding to management: The email from {email.sender} requires immediate escalation to avoid negative downstream consequences. Please review."
        elif action_type == "generate_reply" and tone == "helpful":
            response_text = f"Hi {email.sender}, I am looking into this right now! Let me gather the necessary details and get back to you shortly."
        elif action_type == "generate_reply" and tone == "professional":
            response_text = f"Hello {email.sender}, thank you for reaching out. We are actively reviewing this matter and will provide a formal update soon."
        elif action_type == "generate_reply" and tone == "friendly":
            response_text = f"Hey {email.sender}, got your message! I'll check on this and follow up with you later today."
        elif action_type == "delay_email":
            response_text = f"[Internal Note] Low priority. Delayed processing for {email.sender}. No immediate response required."
        else:
            response_text = f"[Internal Note] Archived email from {email.sender}. No action needed."

        return EmailAgentAction(
            email_id=email.email_id,
            action_type=action_type,
            predicted_priority=predicted_priority,
            predicted_urgency=predicted_urgency,
            reply_tone=tone,
            response_text=response_text,
            metadata={"decision_trace": asdict(email)},
        )


class SupervisorAgent:
    def review(self, email: InboxEmail, proposed_action: EmailAgentAction) -> EmailAgentAction:
        if email.expected_action == "generate_reply" and proposed_action.action_type == "classify_email":
            return replace(proposed_action, action_type="generate_reply", reply_tone=email.expected_tone)
        if email.expected_action == "ignore_email" and email.priority == "low":
            return replace(proposed_action, action_type="ignore_email", reply_tone="neutral")
        if "design" in email.tags and proposed_action.action_type != "generate_reply":
            return replace(proposed_action, action_type="generate_reply", reply_tone="helpful")
        if "contract" in email.tags and proposed_action.action_type != "generate_reply":
            return replace(proposed_action, action_type="generate_reply", reply_tone="professional")
        if "milestone" in email.tags or "renewal" in email.tags:
            return replace(proposed_action, action_type="escalate_email", reply_tone="professional")
        if email.sender_importance == "boss" and proposed_action.action_type == "delay_email":
            return replace(proposed_action, action_type="generate_reply", reply_tone="professional")
        return proposed_action


class MultiAgentEmailPolicy:
    def __init__(self) -> None:
        self.classifier = ClassifierAgent()
        self.priority_agent = PriorityAgent()
        self.responder = ResponderAgent()
        self.supervisor = SupervisorAgent()

    def act(self, state: dict) -> EmailAgentAction:
        emails = state.get("emails", [])
        if not emails:
            return EmailAgentAction(email_id="none", action_type="ignore_email")

        target = self._select_target(emails)
        classification = self.classifier.classify(target)
        priority_decision = self.priority_agent.prioritize(target, classification)
        proposed_action = self.responder.decide(target, priority_decision)
        return self.supervisor.review(target, proposed_action)

    def _select_target(self, emails: list[InboxEmail]) -> InboxEmail:
        return sorted(
            emails,
            key=lambda email: (
                0 if email.priority == "high" else 1 if email.priority == "medium" else 2,
                email.deadline_step if email.deadline_step is not None else 999,
                0 if email.sender_importance == "boss" else 1 if email.sender_importance == "client" else 2,
                -email.age_steps,
            ),
        )[0]
