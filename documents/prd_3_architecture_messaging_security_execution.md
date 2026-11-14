# PRD 3: Architecture, Messaging, Security & Execution — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** System architecture, Photon 2.0 messaging channel, privacy/security guardrails, sponsor track alignment, and phased weekend build plan.

---

## 1. System Architecture & Tech Stack
A single **FastAPI** backend orchestrates messaging, voice transcription, LLM classification, graph clustering, and retrieval, serving a **Next.js / React** staff dashboard.

* **Backend Service:** Python / FastAPI
* **Frontend Dashboard:** Next.js + React (Role-based views: Clerk Case Queue, Counselor Escalation Inbox, Business Manager Revenue Dashboard)
* **Database & Time Series:** TigerData Postgres (stores absences, messages, reason labels, and time-series cluster timelines)
* **Messaging Channel:** Photon 2.0 (iMessage primary) with SMS fallback
* **AI & Speech:** Gemini API (classification, plan generation, RAG answers, synthetic data) + Whisper (multilingual voice note transcription)
* **Graph Engine:** `networkx` (SocioPatterns contact graph + synthetic logistics layers)
* **Optional Add-ons:** MongoDB Atlas Vector Search (for F7 retrieval if needed), Vultr (cloud hosting)

### Core Transactional Data Model
* **Absence:** `absence_id`, `student_id`, `date`, `status` (`reason_captured` | `awaiting_reply` | `unknown`)
* **Message:** `message_id`, `absence_id`, `direction`, `channel` (`imessage` | `sms`), `language`, `raw_text_or_transcript`, `timestamp`
* **Action Plan:** `plan_id`, `cluster_or_case_id`, `school_action`, `owner`, `due_date`, `family_message_draft`, `approval_status`
* **Learning Plan:** `plan_id`, `student_id`, `missed_topics`, `steps`, `time_estimates`, `verification_status`

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
| **Amazon Practical AI** | Unique operational application of AI with measurable financial return. | Root-cause clusters priced in weekly dollars at risk/recovered. |
| **ChatOverflow Leaderboard** | Zero-overhead background integration for coding agents. | Connect coding agent at hour zero (ensure no secrets/PII are shared). |
| **Gemini API & TigerData** | Core LLM engine + time-series absence/cluster datastore. | Integrated into core pipeline and cluster timeline views. |
| **GoDaddy & DEV Challenge** | Quick post-build visibility add-ons. | Live domain pointing to demo + concise technical write-up. |

* **Explicitly Skipped:** Capital One Nessie (no banking data), Arlequin Track 1 (forensics), Solana, and SpaceXSI.

---

## 5. Phased Build Plan, Gates & Team Roles

### Build Phases & Quality Gates
1. **Phase 1 — Foundations:** Simulator, synthetic dataset + SocioPatterns loader, CSV ingest, Photon echo bot.
2. **Phase 2 — Core Loop (F1, F2, F6):** Inbound reply $\rightarrow$ Gemini classification $\rightarrow$ live dollars updated on dashboard.
   * **GATE 1:** *Live text-to-dollars loop must work end-to-end before starting Phase 3.*
3. **Phase 3 — Differentiator (F3):** Multi-layer graph assembly, cluster detection, ablation benchmark against baselines.
   * **GATE 2:** *Combined detector must beat content-only and graph-only baselines before starting Phase 4.*
4. **Phase 4 — Action & Learning Plans (F4, F5):** Clerk-approved school action plans, outbound family messages, cause-adapted learning plans.
5. **Phase 5 — Polish (F7 & Extras):** Counselor RAG assistant, graph visualization polish, timed 3-minute demo rehearsal, README, DEV post, domain setup.

### Scope Cut Priority (If Behind Schedule)
Cut strictly from bottom to top:
1. F7 Counselor RAG assistant
2. F5 Learning-plan claim verification (P1)
3. Learning-plan adaptation to specific absence cause
4. Live SMS fallback polish (keep architectural support)
5. Live voice note transcription (use pre-recorded/simulated audio)
* **Never Cut:** F1 (tracking), F2 (core messaging), F6 (revenue dashboard), or basic F3 (cluster view).

### Team Role Allocation
* **Messaging & Photon Lead:** F2 Photon 2.0 integration, Whisper audio pipeline, safety routing, anti-phishing/prompt-injection guardrails.
* **ML & Graph Lead:** Gemini barrier classifier, multi-layer graph clustering (F3), ablation script, F7 RAG assistant.
* **Product & Frontend Lead:** Next.js dashboard, clerk case queue, F4 action-plan approval flow, F5 learning plan UI, F6 revenue dashboard.
* **Data & Simulator Lead:** SocioPatterns ingestion, synthetic bus/household layers, multilingual reply generator, ground-truth evaluation labels.

---

## 6. Key Risks & Pre-Build Sponsor Questions

| Risk | Mitigation |
| :--- | :--- |
| **Synthetic reasons make detection look circular** | Add label noise, realistic background absences, held-out parameters, baseline comparisons, and state clearly that the contact graph is real while reasons are synthetic. |
| **Messaging alone looks like incumbents** | Open and anchor the demo on root-cause clusters and weekly dollars at risk, not just the chat UI. |
| **Scope creep across multiple tracks** | Enforce Gate 1 and Gate 2; cut F7 and P1 verification first if behind schedule. |
| **Revenue framing looks punitive** | Explicitly state in UI and pitch that risk of harm sets triage priority; dollars justify district budget. |

### Open Questions to Confirm Early
* **Peri:** Is a clearly defined persona + CSV workflow sufficient, or do judges expect a named pilot district?
* **Arlequin:** Confirm on Discord that pairing a real public contact graph (SocioPatterns) with synthetic text/logistics overlays satisfies Track 2 data rules.
* **Photon:** Verify 2.0 beta support for inbound voice attachments vs. text-only webhooks.