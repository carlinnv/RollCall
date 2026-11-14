# PRD 3: Architecture, Messaging, Security & Execution — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** System architecture, Photon 2.0 messaging channel, privacy/security guardrails, sponsor track alignment, and phased weekend build plan.

---

## 1. System Architecture & Tech Stack
A single **FastAPI** backend orchestrates messaging, voice transcription, LLM classification, graph clustering, agent actions, and retrieval, serving a **React (Vite)** staff dashboard.

* **Backend Service:** Python / FastAPI
* **Frontend Dashboard:** React 18 + TypeScript + Vite (Role-based views: Clerk Morning Briefing & Case Queue, Counselor Escalation Inbox, Business Manager Revenue Dashboard)
* **Database & Time Series:** TigerData Postgres (stores absences, messages, reason labels, clerk corrections, and time-series cluster timelines; continuous aggregates for weekly dollars; pgvector for F7 retrieval). Replaces the current in-memory store.
* **Messaging Channel:** Photon 2.0 (iMessage primary) with SMS fallback
* **AI & Speech:** Gemini API (classification, plan generation, RAG answers, synthetic data) + Whisper (multilingual voice note transcription)
* **Graph Engine:** `networkx` (SocioPatterns contact graph + synthetic logistics layers)
* **Hosting:** Vultr (backend + dashboard, public demo URL)
* **Optional Add-ons:** LangChain Deep Agents (F7 counselor assistant), Amazon Bedrock (one component, e.g. safety classifier or RAG answers)

### Core Transactional Data Model
* **Absence:** `absence_id`, `student_id`, `date`, `status` (`reason_captured` | `awaiting_reply` | `unknown`)
* **Message:** `message_id`, `absence_id`, `direction`, `channel` (`imessage` | `sms`), `language`, `raw_text_or_transcript`, `timestamp`
* **Action Plan:** `plan_id`, `cluster_or_case_id`, `school_action`, `owner`, `due_date`, `family_message_draft`, `approval_status`
* **Learning Plan:** `plan_id`, `student_id`, `missed_topics`, `steps`, `time_estimates`, `verification_status`
* **Agent Action:** `action_id`, `cluster_or_case_id`, `type` (`transport_alert` | `nurse_notice` | `family_followup` | `counselor_task`), `recipient`, `draft`, `approval_status`
* **Label Correction:** `absence_id`, `old_label`, `new_label`, `corrected_by`, `timestamp`

---

## 2. Messaging Agent & Channel Requirements (F2 — P0)
* **Trigger:** Automatically sends an initial message after a student's first recorded absence, stating their current absence count and asking one open, supportive question about what got in the way.
* **Multimodal & Multilingual:** Accepts text replies and audio voice notes in any language; transcribes voice via Whisper and processes end-to-end within **10 seconds**.
* **Channel Resilience:** Delivers via Photon 2.0 iMessage with automatic SMS fallback so Android households are never excluded.
* **Acceptance:** A live Spanish reply during the demo produces a structured barrier label within 10 seconds, and a seeded safety message routes immediately to the counselor inbox.

---

## 3. Privacy, Security & Ethics Guardrails

### Messaging Channel Security
* **Anti-Phishing / Spoofing:** Use a verified sender identity and include a district verification code families can check. **Never include URLs/links**, and never request money, passwords, or ID numbers.
* **Prompt-Injection Defense:** Treat all inbound family replies strictly as untrusted data. Constrain Gemini outputs to strict JSON schemas; never allow message content to trigger tools or execute instructions.
* **Compliance & Auditability:** Enforce rate limits, honor `STOP`/opt-out keywords immediately, and maintain an immutable audit log of every message, classification, and human approval.
* **Human-in-the-Loop:** A human staff member must approve every outbound action plan, and 100% of safety-flagged messages bypass automation straight to a counselor.

### Data Privacy & Ethical Guardrails
* **Zero Real Student PII:** Use only public anonymized data (SocioPatterns) and synthetic student/family records.
* **Role-Based Access & Aggregation:** Clerks manage individual cases; counselors manage escalations; business managers see financial aggregates. Cluster views display group patterns rather than individual names unless direct intervention is required.
* **No Friendship Inference:** Use the contact network strictly for aggregate cluster detection; never infer or expose student social relationships.
* **Equity Auditing:** Report barrier classification accuracy across languages to ensure non-English-speaking families receive equal service quality.

---

## 4. Hackathon Track & Sponsor Alignment

| Track / Sponsor | Fit & Strategy | Demo Deliverable Required |
| :--- | :--- | :--- |
| **Education (Main Track)** | Addresses chronic absenteeism, personalized catch-up plans, and AI trust safeguards. | Live flow from absence $\rightarrow$ reason $\rightarrow$ root cause $\rightarrow$ learning plan. |
| **Peri Forward-Deployed** | Solves 1 concrete issue (unknown absence reasons) for 1 persona with clear dollar ROI. | Named clerk/CFO workflow, dollar metric, deployable CSV-import setup. |
| **Photon Messaging Agent** | Two-way family communication is the core ingestion loop. | Live Photon 2.0 iMessage conversation + SMS fallback architecture. |
| **Arlequin Track 2** *(Content + Connectivity)* | Combines message semantics with a school proximity/transit contact graph. | Ablation table, interactive graph visualization, Recall@k metric. |
| **Amazon Practical AI** | An agent that reasons (root cause), plans (action plan), acts (drafted alerts and messages), and adapts (learns from clerk corrections), used daily by a clerk. | Morning briefing $\rightarrow$ cluster $\rightarrow$ one-click approved actions; show a label correction changing the next classification. |
| **LangChain Deep Agents** *(optional)* | F7 counselor assistant built on Deep Agents with sub-agents and a file system. | Counselor asks a question; agent returns a cited case report. |
| **ChatOverflow Leaderboard** | Zero-overhead background integration for coding agents. | Connect coding agent at hour zero (ensure no secrets/PII are shared). |
| **Gemini API & TigerData** | Core LLM engine + time-series absence/cluster datastore with pgvector. | Integrated into core pipeline and cluster timeline views. |
| **Vultr, GoDaddy & DEV Challenge** | Quick hosting and visibility add-ons. | Demo hosted on Vultr behind a live domain + concise technical write-up. |

* **Explicitly Skipped:** Capital One Nessie (no banking data), Arlequin Track 1 (forensics), CoStar (real-estate data), Best Physical AI Hack (needs hardware), MongoDB Atlas (TigerData chosen instead), Solana, and SpaceXSI (requires Cursor + Grok Bot).

---

## 5. Phased Build Plan, Gates & Team Roles

### Build Phases & Quality Gates
1. **Phase 1 — Foundations:** Simulator, synthetic dataset + SocioPatterns loader, CSV ingest, Photon echo bot.
2. **Phase 2 — Core Loop (F1, F2, F6):** Inbound reply $\rightarrow$ Gemini classification $\rightarrow$ live dollars updated on dashboard.
   * **GATE 1:** *Live text-to-dollars loop must work end-to-end before starting Phase 3.*
3. **Phase 3 — Differentiator (F3):** Multi-layer graph assembly, cluster detection, ablation benchmark against baselines.
   * **GATE 2:** *Combined detector must beat content-only and graph-only baselines before starting Phase 4.*
4. **Phase 4 — Actions & Learning Plans (F4, F5, F8, F9):** Clerk-approved school action plans and agent actions, outbound family messages, morning briefing, label-correction feedback loop, cause-adapted learning plans.
5. **Phase 5 — Polish (F7 & Extras):** Counselor RAG assistant, graph visualization polish, Vultr deploy, timed 3-minute demo rehearsal with pre-generated fallback data, README, DEV post, domain setup.

### Scope Cut Priority (If Behind Schedule)
Cut strictly from bottom to top:
1. F7 Counselor RAG assistant (and the optional Deep Agents version)
2. F5 Learning-plan claim verification (P1)
3. F8 learning from corrections (P1; keep the drafted actions)
4. Learning-plan adaptation to specific absence cause
5. Live SMS fallback polish (keep architectural support)
6. Live voice note transcription (use pre-recorded/simulated audio)
* **Never Cut:** F1 (tracking), F2 (core messaging), F6 (revenue dashboard), basic F3 (cluster view), or F9 (morning briefing).

### Team Role Allocation
* **Messaging & Photon Lead:** F2 Photon 2.0 integration, Whisper audio pipeline, safety routing, anti-phishing/prompt-injection guardrails.
* **ML & Graph Lead:** Gemini barrier classifier, correction feedback loop (F8), multi-layer graph clustering (F3), ablation script, F7 RAG assistant.
* **Product & Frontend Lead:** React dashboard, morning briefing (F9), clerk case queue, F4 action-plan and F8 action approval flow, F5 learning plan UI, F6 revenue dashboard.
* **Data & Simulator Lead:** SocioPatterns ingestion, synthetic bus/household layers, multilingual reply generator, ground-truth evaluation labels.

---

## 6. Key Risks & Pre-Build Sponsor Questions

| Risk | Mitigation |
| :--- | :--- |
| **Synthetic reasons make detection look circular** | Add label noise, realistic background absences, held-out parameters, baseline comparisons, and state clearly that the contact graph is real while reasons are synthetic. |
| **Messaging alone looks like incumbents** | Open and anchor the demo on root-cause clusters and weekly dollars at risk, not just the chat UI. |
| **Scope creep across multiple tracks** | Enforce Gate 1 and Gate 2; cut F7 and P1 verification first if behind schedule. |
| **Revenue framing looks punitive** | Explicitly state in UI and pitch that risk of harm sets triage priority; dollars justify district budget. |
| **Live API failure during demo** | Keep pre-generated simulated data and a scripted demo path that works without live Gemini or Photon calls. |

### Open Questions to Confirm Early
* **Devpost:** How many sponsor tracks can one team enter? Do MLH prizes stack with sponsor tracks?
* **Peri:** Is a clearly defined persona + CSV workflow sufficient, or do judges expect a named pilot district?
* **Arlequin (high priority):** Track 2 says "public data only" (Track 1 allows synthetic). Confirm on Discord that a real public contact graph (SocioPatterns) with synthetic replies and logistics layers is acceptable before building on it.
* **Photon:** Verify 2.0 beta support for inbound voice attachments vs. text-only webhooks.