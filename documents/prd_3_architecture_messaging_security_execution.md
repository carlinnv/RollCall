# PRD 3: Architecture, Messaging, Security & Execution — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** System architecture, Photon 2.0 messaging channel, privacy/security guardrails, sponsor track alignment, and phased weekend build plan.

---

## 1. System Architecture & Tech Stack
A single **FastAPI** backend orchestrates messaging, voice, LLM classification, graph clustering, agent actions, and retrieval, serving a **React (Vite)** staff dashboard.

* **Backend Service:** Python 3.11/3.12 + FastAPI, single uvicorn worker (the in-memory store has no locking). Python 3.14 is not supported by the pinned pydantic.
* **Frontend Dashboard:** React 18 + TypeScript + Vite. Role-based views: Clerk Morning Briefing & Case Queue, Counselor Escalation Inbox, Liaison Inbox, Business Manager Revenue Dashboard. Polls (or uses SSE) so a live reply appears without reloading.
* **Source of truth:** the in-memory store, loaded from a precomputed snapshot (labels included) so `reset()` makes no LLM calls. Clusters are cached behind a version counter that bumps on every change.
* **TigerData (sidecar):** one hypertable `absence_events` plus one continuous aggregate `weekly_dollars_by_cause` driving the revenue/early-warning timeline chart. Optional pgvector for F7. Not a full migration; the app keeps working if TigerData is down.
* **Messaging Channel:** Photon 2.0 (iMessage primary) with SMS fallback; a phone-number index maps inbound messages to the student and open absence.
* **AI & Speech:** Gemini API (classification, native audio for voice notes, plan generation, assistant answers). Whisper only as a fallback. Synthetic replies are generated with a different model family than the classifier (PRD 2 §2).
* **Graph Engine:** `networkx` + NumPy (SocioPatterns contact graphs + synthetic logistics layers).
* **Hosting:** Vultr with HTTPS from hour zero (the Photon webhook needs a public URL).
* **Optional Add-ons:** LangChain Deep Agents (F7 counselor assistant); Amazon Bedrock for one component (e.g., the safety classifier).
* **Demo mode (`DEMO_MODE`):** cached model outputs, a replayable webhook fixture, and a Photon-off path, so the scripted demo runs with no network.

### Core Transactional Data Model
* **Absence:** `absence_id`, `student_id`, `date`, `status` (`reason_captured` | `awaiting_reply` | `unknown`), `suggested_sis_code`
* **Message:** `message_id`, `absence_id`, `direction`, `channel` (`imessage` | `sms` | `voice`), `language`, `raw_text_or_transcript`, `timestamp` (real `now()`), `latency_ms`. An absence can have many messages.
* **Action Plan:** `plan_id`, `cluster_or_case_id`, `school_action`, `owner`, `approver_role`, `due_date`, `family_message_draft`, `approval_status`
* **Agent Action:** `action_id`, `cluster_or_case_id`, `type` (`recode_flag` | `transport_notice` | `nurse_notice` | `family_followup` | `counselor_task`), `recipient`, `draft`, `approval_status`
* **Learning Plan:** `plan_id`, `student_id`, `missed_topics`, `steps`, `time_estimates`, `verification_status`
* **Escalation:** `escalation_id`, `student_id`, `household_id`, `opened_at`, `acknowledged_at`, `acknowledged_by`, `report_status` (`report_filed` | `not_required`), `report_by`, `report_at`, `closed_at`
* **Label Correction:** `absence_id`, `old_label`, `new_label`, `corrected_by`, `timestamp`
* **Audit Entry:** append-only, hash-chained record of every message, classification, model call, approval and send.

---

## 2. Messaging Agent & Channel Requirements (F2 — P0)
* **Trigger:** if no reason is on file, send one supportive check-in after an absence asking what got in the way, and saying who reads replies. No absence counts in the first message (count-based messages follow PRD 1 Outreach Rules).
* **Multimodal & Multilingual:** accepts text and voice notes in any language; processes end-to-end within **10 seconds** (measured with real timestamps). Messages go out in the family's language (Ed Code 48985); machine-translated templates are reviewed by a human before use.
* **Channel Resilience:** Photon 2.0 iMessage with automatic SMS fallback so Android households are never excluded.
* **Acceptance:** a live Spanish reply produces a structured barrier label within 10 seconds, and a seeded safety message routes immediately to the counselor inbox. Both also work with Photon off (demo mode).

---

## 3. Privacy, Security & Ethics Guardrails

### Safety Escalation Process
* **An open safety escalation suppresses all automated and drafted outbound messages to that student and household until a counselor closes it.** This applies to case plans, cluster plans, check-ins and follow-ups. Only the counselor contacts the family.
* **Owner and timing:** safety flags page the on-call counselor; acknowledge within 15 minutes during school hours. After hours, page the on-call administrator; escalate to the principal if unacknowledged. A human acknowledges the family within 4 school hours.
* **Mandated reporting:** the system is not the reporter. Staff who read a disclosure follow CANRA (PC 11166): immediate phone report, written report within 36 hours. The inbox records `report_filed / not_required` with who and when. Bullying also goes to site admin under the district bullying policy.
* **Automated text after a flag:** only one fixed, district-approved crisis-resource message (988/911), never LLM text, and only for self-harm or imminent danger. Never for abuse disclosures, where the guardian's phone may not be safe.

### Messaging Channel Security & Compliance
* **Anti-phishing:** verified sender identity; never include URLs/links; never request money, passwords, or ID numbers. Instead of a static verification code (which a phisher can copy): *"We never ask for money, passwords or SSNs. To check this message is from us, call the office number on the school website."*
* **Consent & opt-out:** message only the guardian number on file. Honor STOP and plain-language opt-outs in English, Spanish and Vietnamese ("no más mensajes") immediately. Quiet hours 8am–8pm local; queue otherwise. SMS via a registered 10DLC or short code.
* **Prompt-injection defense:** treat all inbound replies as untrusted data. Outputs constrained to JSON schemas; labels must match the enum; `evidence_quote` must be an exact substring of the input. Message content never triggers tools. LLM output can only add a safety flag, never clear one. Only staff-confirmed corrections become examples. The F7 agent has read-only tools and no send.
* **Auditability:** append-only, hash-chained audit log of every message, classification, model call, approval and send, viewable in an Audit tab.
* **Human-in-the-loop:** the owning role approves every outbound plan and action; 100% of safety-flagged messages bypass automation.

### Data Privacy & Ethical Guardrails
* **Zero Real Student PII:** public anonymized contact graphs (SocioPatterns) plus synthetic student/family records. The UI shows a "Synthetic data — fictional school" banner; seed names should be obviously fictional.
* **Model calls:** replace names with placeholders before any LLM call; show a "sent to model" panel in the demo. Production: Vertex AI or Bedrock under a DPA (no training, zero retention, US region); contracts meet SOPIPA and AB 1584 (district owns the data, no ads or profiling, deletion on request, breach notice). Whisper, if used, is self-hosted.
* **Role-Based Access:** auth with roles (demo login or role switcher). Clerks see "Safety: with counselor", never the disclosure text. Housing details go to the McKinney-Vento liaison only. Business-manager views are aggregates with cells of n < 11 suppressed. `/api/reset` is disabled in the public build.
* **Minimum necessary in agent actions:** transport notice = route, dates, rider count (no names, replies or barriers; the transport office may be a contractor). Nurse notice = class section, dates, count, symptom category; names are opened in the dashboard. Counselor tasks stay inside the app.
* **Contact graph:** used only for illness-spread scoring, shown collapsed to class-section level with no student-to-student edges. Node IDs re-keyed; original class names dropped. Respect CC BY-NC-SA (non-commercial). In production we would not collect proximity data; class schedules serve the same purpose.
* **No friendship inference; no immigration status.** The `community safety concern` label stores no free-text details.
* **Equity Auditing:** report classification accuracy per language so non-English-speaking families receive equal service.

---

## 4. Hackathon Track & Sponsor Alignment

| Track / Sponsor | Fit & Strategy | Must Show on Stage |
| :--- | :--- | :--- |
| **Main judging** | Social need (chronic absence), creative cross-family root cause, working loop. | Reply $\rightarrow$ cluster $\rightarrow$ approved fix in 3 minutes; close on kids, not dollars. |
| **Amazon Practical AI** | An agent that reasons (root cause), plans, acts (drafted recode flags, notices, messages), and adapts (learns from corrections), used daily by a clerk. | Briefing $\rightarrow$ drafted actions $\rightarrow$ one-click approve $\rightarrow$ a correction changes the next result. |
| **Arlequin Track 2** *(Content + Connectivity)* | Message semantics + school contact/transit graph; both layers measured. | Held-out ablation (recall at equal false-alarm rate, lead time), interactive aggregate graph view, Recall@5. Confirm data rules with Arlequin early. |
| **Gemini API (MLH)** | Core LLM: native audio + structured JSON + assistant answers. | Voice note $\rightarrow$ label in one Gemini call; per-language F1. |
| **TigerData (MLH)** | Hypertable + continuous aggregate for absence events. | The "flagged N days earlier" timeline driven by the continuous aggregate. |
| **LangChain Deep Agents** | F7 counselor assistant with sub-agents and a file system. | Counselor asks a question; agent returns a cited case report. |
| **Photon Messaging Agent** | Two-way family communication is the core ingestion loop. | Live Photon 2.0 conversation + SMS fallback; simulated channel as backup. |
| **Peri Forward-Deployed** | One concrete issue (unknown absence reasons) for one persona with dollar impact. | Named clerk workflow, CSV import, aggregate dollar metric. |
| **ChatOverflow Leaderboard** | Zero-overhead background integration for coding agents. | Connect coding agent at hour zero (no secrets/PII shared). |
| **Vultr, GoDaddy & DEV Challenge** | Hosting and visibility add-ons. | Demo hosted on Vultr behind a live domain + technical write-up. |

* **Explicitly Skipped:** Capital One Nessie (no banking data), Arlequin Track 1 (forensics), CoStar (real-estate data), Best Physical AI Hack (needs hardware), MongoDB Atlas (TigerData chosen instead), Solana, and SpaceXSI (requires Cursor + Grok Bot).
* **One story for every track:** *"An attendance agent that finds why students are absent, traces it to a shared root cause by combining what families say with the school's real contact network, and acts on it."* Shift emphasis per judge.

---

## 5. Phased Build Plan, Gates & Team Roles
Owners: **A** = Messaging & Photon, **B** = ML & Graph, **C** = Product & Frontend, **D** = Data & Simulator. T0 = start of build.

1. **Phase 0 (T0–2), all:** freeze data contracts (Message, Cluster v2, AgentAction, Correction, Escalation); add `DATASET=demo|sim`; deploy a skeleton to Vultr with HTTPS; connect ChatOverflow; ask Arlequin and Photon questions on Discord.
2. **Phase 1 (T2–10):**
   * **A:** Photon spike with go/no-go at T5; then webhook, phone index, outbound sends, STOP handling.
   * **B:** Gemini structured-output classifier (6 s timeout, keyword fallback, cache); fix the safety gate and keyword bugs.
   * **C:** morning briefing, agent-action approve/edit/dismiss, live polling, loading states.
   * **D:** simulator on the Lyon and Marseille graphs with fixed seeds, planted events, reply delay/missingness, ground truth; batch-generate and cache replies; with C, write the hand-labeled set.
   * **GATE 1 (T10):** a phone text shows label + evidence on the dashboard in ≤ 10 s, and the same flow works with Photon off.
3. **Phase 2 (T10–16):**
   * **B:** permutation scan statistic, three detectors, held-out ablation (`eval/ablation.json`).
   * **C:** aggregate graph view and ablation table.
   * **A:** voice notes through Gemini audio.
   * **GATE 2 (T16):** the combined detector beats both baselines on held-out runs; otherwise report honestly and stop tuning at T18.
4. **Phase 3 (T16–22):**
   * **B:** F8 learning from corrections; then F7 Deep Agents assistant (own time box, ≤ 3 h).
   * **C:** label-correction UI, role switcher, audit tab.
   * **D:** TigerData sidecar (hypertable + continuous aggregate + timeline chart); scaling benchmark chart.
   * **A:** consent/quiet-hours/opt-out polish; escalation SLA timer.
5. **Freeze (T22).** T22–26: `DEMO_MODE`, three timed rehearsals, backup video. T26–30: Devpost, README, DEV post, domain.

### Scope Cut Priority (If Behind Schedule)
Cut from the top:
1. F5 claim verification (P1)
2. pgvector (F7 keeps keyword/SQL retrieval)
3. Whisper fallback
4. Scaling benchmark chart
5. Live SMS fallback polish (keep architectural support)
6. p-values (fall back to rule-based clusters, labeled as such)
* **Track work is time-boxed, not cut:** F7 Deep Agents (≤ 3 h, after Gate 2) and the TigerData sidecar (≤ 3 h, after Gate 1) each have an owner and must not block the core demo.
* **Never Cut:** F1, F2 text loop, safety gate, basic F3, F6, F8 drafted actions and corrections, F9, `DEMO_MODE`.

---

## 6. Key Risks & Pre-Build Sponsor Questions

| Risk | Mitigation |
| :--- | :--- |
| **Synthetic reasons make detection look circular** | Simulator mechanisms the detector does not encode, reply delay, held-out config and graph, label-noise curve, hand-labeled set; state clearly that the contact graph is real and the replies are synthetic. |
| **Messaging alone looks like incumbents** | Anchor the demo on the cross-family root cause, not the chat UI. |
| **Scope creep across many tracks** | Gates 1 and 2; each extra track has an owner and a time box after the core gates. |
| **Revenue framing looks punitive** | Dollars in aggregate only, never per student; close on kids; risk of harm sets priority. |
| **Live API or Photon failure during demo** | `DEMO_MODE` with cached outputs and a replayable fixture; backup video. |
| **Gemini latency/rate limits** | Precomputed labels on reset, batching, caching, 6 s timeout with keyword fallback. |
| **Safety miss or wrong-role exposure** | Household-level suppression, keyword check that the model can't clear, fail-closed, RBAC, audit log. |

### Open Questions to Confirm Early
* **Team:** rename to "Here"? (See PRD 1.)
* **Arlequin (high priority):** Track 2 says "public data only" (Track 1 allows synthetic). Confirm that a real public contact graph with synthetic replies and logistics layers is acceptable, and that school–family messages count as "social" data for this track.
* **Peri:** is a clearly defined persona + CSV workflow sufficient, or do judges expect a named pilot district?
* **Photon:** does 2.0 support inbound voice attachments, and does it need a macOS host with Messages?
* **Facts to verify before the pitch:** CA 2024–25 chronic absence rate; Attendance Recovery rules (start year, 10-day cap, supervision ratio); Utah dataset license.
