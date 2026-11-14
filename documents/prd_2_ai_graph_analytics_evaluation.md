# PRD 2: AI, Graph Analytics & Evaluation — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** ML/LLM pipelines, contact + logistics graph modeling, root-cause cluster detection, ablation methodology, and RAG assistant retrieval.

---

## 1. Core AI Thesis
**Hypothesis:** Combining semantic barrier labels extracted from family messages (**Content Layer**) with physical/logistical school networks (**Connectivity Layer**) detects actionable root causes earlier and more accurately than analyzing either layer in isolation.

---

## 2. Data Sources & Synthetic Simulation

### Datasets
* **Contact Graph (Real & Public):** SocioPatterns high school dataset (~329 students across 9 classes, face-to-face proximity contacts measured over 5 days in Dec 2013; CC BY-NC-SA 3.0 — cite *Mastrandrea et al., PLoS ONE 2015*). *(Note: Evaluate SocioPatterns primary/elementary datasets for K-12 alignment and licensing).*
* **Logistics Layers (Synthetic):** Assign synthetic bus routes, household IDs, and neighborhoods to every student node.
* **Absences & Replies (Synthetic):** A simulator propagates contagion-like illness along proximity contact edges, triggers bus-route failures on specific days, and injects realistic background noise. Gemini generates multilingual family replies containing typos, terse phrasing, and label noise, each tagged with a ground-truth barrier label for evaluation.

### Core Data Entities (ML & Graph Scope)
* **Student:** `anonymous_id`, `grade`, `class_sections`, `bus_route`, `household_id`, `neighborhood`
* **Reason:** `absence_id`, `barrier_label`, `confidence_score`, `evidence_quote`, `source_message_id`
* **Cluster:** `layers_involved`, `member_ids`, `likely_cause`, `start_date`, `weekly_revenue_at_risk`, `evidence_message_ids`

---

## 3. AI & Graph Pipeline Specifications

### 3.1 Multilingual Voice & Reason Classification (F1 / F2)
* **Speech-to-Text:** Pass inbound voice notes through **Whisper** to produce a verbatim transcript while preserving original language.
* **Barrier Classification:** **Gemini** maps the text/transcript to one label from the 8-class taxonomy (`illness`, `transportation`, `housing or food insecurity`, `caregiving or family obligation`, `safety or bullying`, `disengagement or academic struggle`, `appointment or travel`, `unknown`).
* **Structured Output:** Must return constrained JSON: `{ "label": string, "confidence": float, "evidence_quote": string }`.
* **Fallback:** Replies below confidence threshold or labeled `unknown` are routed to the clerk queue for manual review.

### 3.2 Root-Cause Cluster Detection (F3 — P0)
* **Multi-Layer Graph Construction:** Combine the real proximity contact graph with synthetic class section, bus route, household, and neighborhood edges (using `networkx`).
* **Statistical Over-Representation Scan:** For each barrier type, scan connected subgraphs within each layer and test whether the barrier frequency is statistically over-represented relative to baseline chance over the same time window.
* **Cause Attribution Rules:**
  * Barrier `illness` spreading across proximity contact edges over consecutive days $\rightarrow$ **Contagion-like illness cluster**.
  * Barrier `transportation` concentrated on a single route $\rightarrow$ **Bus/route failure**.
* **Output:** Rank clusters by statistical strength and student count; attach cited source messages and calculated weekly revenue at risk.
* **Acceptance:** Demo surfaces at least one illness cluster and one route cluster, each backed by cited family messages.

### 3.3 Grounded Plan Generation & Verification (F4 / F5)
* **Structured Generation:** Generate action plans and student learning plans strictly from structured inputs (cluster metadata, cause, course pacing guide, missed dates) rather than open-ended free text.
* **Claim Verification (F5 — P1):** Check every generated learning-plan topic and statement against the course pacing guide and source materials using the NLI/claim-verification pipeline; flag any unsupported claims.

### 3.4 Counselor RAG Assistant (F7 — P1)
* Embed messages and structured reasons into a vector index (TigerData pgvector or MongoDB Atlas Vector Search).
* Retrieve top-$k$ relevant messages for natural-language staff queries (e.g., *"Why are students on Route 12 missing school this week?"*).
* Constrain Gemini to answer **only** from retrieved context with explicit message citations.

### 3.5 Safety Triage Classifier (F2 — P0)
* Dedicated safety check running across all inbound languages to detect abuse, neglect, bullying, or threats of self-harm/violence.
* Immediately flags the case for human counselor intervention and suppresses all automated outbound replies.

---

## 4. Evaluation Framework & Target Metrics

| Metric | Target | Measurement Methodology |
| :--- | :--- | :--- |
| **Barrier Classification** | Macro-F1 $\ge 0.80$ | Evaluated on held-out labeled synthetic replies (including noisy and multilingual text). |
| **Cluster Detection** | Outperforms single-layer baselines on Recall at fixed Precision | Ablation study on planted and simulated events. |
| **Early Warning Lead Time** | Flags cluster $\ge 2$ days earlier than raw absence-count thresholds | Replay of simulated multi-day outbreak/route timelines. |
| **Assistant Retrieval (F7)** | Recall@5 $\ge 0.85$ | Evaluated against a labeled question-and-message benchmark set. |
| **Safety Escalation** | **100%** recall on seeded safety-critical messages | Seeded adversarial and multilingual safety test suite. |

### Ablation Study Design (Required for Demo & Arlequin Track 2)
Compare three detectors across identical simulated event runs:
1. **Content-Only Detector:** Tracks barrier counts aggregated by school or grade (Expected weakness: cannot distinguish an active outbreak from random coincidence).
2. **Graph-Only Detector:** Detects connected clusters of absences without reason text (Expected weakness: cannot distinguish a bus breakdown from an illness cluster).
3. **Combined Detector (Ours):** Uses both layers (Expected result: highest recall at equal precision and $\ge 2$ days earlier detection).

### Mitigating Evaluation Circularity
* Because the simulator plants events that the detector finds, we prevent circularity by injecting label noise, realistic background absences, terse/multilingual replies, varying cluster sizes, and testing on held-out simulator parameters while anchoring on a **real** empirical contact network.