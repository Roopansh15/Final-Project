---
title: "InboxWorld: Teaching LLMs to Survive Corporate Triage"
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# 🚀 InboxWorld: Teaching LLMs to Survive Corporate Triage

*Are we really still building chatbots that just summarize text?*

The AI industry is currently obsessed with building simple assistants that grade immediate text output. But in the real world, decision-making is rarely that simple. 

**InboxWorld** targets a critical capability gap: **Long-Horizon Planning in Corporate Triage.** An AI needs to know that if it delays a boring email from the CEO today, it will cause a catastrophic escalation tomorrow. This is an underexplored domain in Reinforcement Learning: teaching an AI to survive a high-stakes, time-sensitive environment where mistakes compound over time.

---

## 🌪️ Phase 1: The "Chaotic" Physics Engine

Most teams build static grid-worlds. We built a time-traveling physics engine based strictly on the OpenEnv standard. 

Real email triage is not just about spotting urgent keywords. A message can sound casual but still block an important deliverable. To model this, InboxWorld introduces:
1.  **A Ticking Clock:** Every action advances the `time_step`. You cannot stall.
2.  **Delayed Consequences:** If the agent ignores a blocking task in Step 1, it isn't punished immediately. Instead, an angry follow-up email is spawned in Step 4. 
3.  **Schema Drift:** Priorities shift based on context (e.g., "The manager is on vacation, so escalate to the VP").

This environment doesn't just grade an LLM on politeness; it grades the LLM on **operational survival.**

---

## 🧠 Phase 2: The Multi-Agent Architecture

To beat this environment, a monolithic prompt isn't enough. We built a **Four-Role Multi-Agent System** that acts as its own step-level verifier.

1.  **Inbox Analyst:** Dynamically extracts tags and intent (e.g., "revenue blocker").
2.  **Priority Planner:** Re-evaluates urgency based on context.
3.  **Responder Agent:** Chooses a rigid structural action (Reply, Draft, Delay, Escalate) and Tone.
4.  **Supervisor Agent:** An intermediate verifier that intercepts bad logic (e.g., *“You cannot delay the CEO. Overriding action to Generate Reply.”*)

**The Reward Matrix:** To prevent reward hacking, we use a multi-factor system. The agent gets immediate lightweight points (+5) for matching the correct tone, but faces massive Delayed Penalties (-10) if an action causes a deadline to be missed three turns later. 

---

## 💻 Phase 3: The SFT Training Pipeline

We didn't just build the environment; we proved that an AI can learn its physics. 

Using **Google Colab** and the **Hugging Face TRL (Transformer Reinforcement Learning)** library, we trained `Qwen/Qwen1.5-0.5B` on curated InboxWorld scenarios.
*   We mapped our Multi-Agent JSON outputs into a massive training corpus.
*   We utilized `SFTTrainer` to teach the model exactly how to map "Sender Importance" and "Tags" to perfect "Action" choices.
*   The loss curve (available in our GitHub repository) proves the model successfully converged, dropping significantly over the training steps.

---

## 🔌 Phase 4: Production LLM Inference (Live Demo)

Training a model is only half the battle; deploying it is the other.

For this Hugging Face Space, we replaced our local heuristic agents with a **Production Inference Pipeline** (`llm_inference.py`). 
*   We utilize the Hugging Face `InferenceClient` to hit the massive `Qwen2.5-72B-Instruct` API via Serverless endpoints.
*   The UI you see below doesn't just rely on keywords. It sends your custom email to a 72-Billion parameter brain.
*   **Try the semantic test:** Type *"we are late, come fast"* into the body. Even though the word "urgent" isn't there, the LLM understands the semantic panic, automatically flags the urgency as `True`, and escalates the thread dynamically.

---

## 📈 The Proof: Observable Improvement

We ran a simulation connecting our evaluation loop directly to the environment. The results were stark:
*   **The Baseline Agent** acted greedily. It achieved a catastrophic score of **-85.0** and missed **56 deadlines** because it possessed no long-horizon foresight.
*   **Our Trained Multi-Agent Policy** mastered the delayed consequences. It surged to a score of **+104.0** and reduced missed deadlines to **Zero**.

*(See our GitHub repository for the full Simulation graphs and Colab Notebook links).*

---

## 🎯 Conclusion

Many AI workflows still reduce email triage to surface-level classification. InboxWorld pushes the frontier by focusing on **operational decision quality** instead. 

A messy, ambitious environment with delayed consequences, real training execution, and live LLM inference beats a polished but boring chatbot every single time. InboxWorld proves that we can teach LLMs not just to speak, but to strategically survive.

