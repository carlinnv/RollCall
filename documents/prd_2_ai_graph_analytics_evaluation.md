# PRD 2: AI, Graph Analytics & Evaluation — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** ML/LLM pipelines, contact + logistics graph modeling, root-cause cluster detection, ablation methodology, and assistant retrieval.

---

## 1. Core AI Thesis
**Hypothesis:** Combining semantic barrier labels extracted from family messages (**Content Layer**) with physical/logistical school networks (**Connectivity Layer**) detects actionable root causes earlier and more accurately than analyzing either layer in isolation.

**Insight to test and report honestly:** SocioPatterns contacts are mostly within a class. Measure what contact ego-nets add *beyond* class sections (e.g., cross-class contagion that section-level zones miss). A null result is still worth reporting.

---

## 2. Data Sources & Synthetic Simulation

### Datasets
* **Contact graphs (real & public):**
  * **Tuning graph:** SocioPatterns Marseille high school 2013 (327 students, 9 classes, 5 days; CC BY-NC-SA 3.0; cite *Mastrandrea et al., PLoS ONE 2015*). Use the face-to-face contact layer only; never the friendship or Facebook layers.
  * **Demo + held-out test graph:** SocioPatterns Lyon primary school 2009 (232 children + 10 teachers, 5 grades × 2 classes, 2 days; cite *Stehlé et al., PLoS ONE 2011*). Elementary, so it matches the demo school.
  * **Optional extra held-out graphs:** Utah Elem1 (339) and Mid1 (591) from *Toth et al., J. R. Soc. Interface 2015*, available via HypergraphX. License not stated there; check the paper's supplementary material before use.
  * **Do not merge schools into one graph.** They have no edges between them and come from different sensors and time resolutions.
* **Daily graph construction:** `w_ij` = seconds of contact per day; keep each node's top-k edges by weight (rank-based, robust across sensors) rather than an absolute threshold. Cycle the recorded days across the simulated calendar.
* **Graph handling rules:** re-key node IDs and drop original class names; contact edges are used only for illness-spread scoring; never attach safety or housing labels to real-graph nodes.
* **Logistics Layers (Synthetic):** Assign synthetic bus routes, household IDs, and neighborhoods to every student node.
* **Scaling benchmark (synthetic):** a 10k–50k-student stochastic block model with class structure matched to the real graphs, used only for a runtime-vs-N chart (Arlequin "responsible compute").

### Simulator
* **Mechanisms the detector does not encode** (to reduce circularity):
  * SEIR illness over the *temporal* contacts with `p_ij = 1 − exp(−β · w_ij)`.
  * Route failures where only 30–80% of riders end up absent.
  * School-wide flu waves that must **not** raise alarms.
  * Per-student absence propensity drawn from a Gamma distribution (chronic absentees).
  * At least one event type no rule targets (e.g., a neighborhood caregiving spike from a daycare closure).
* **Replies:** reply rate 40–90%, reply delay 0–3 days; detectors may only use replies received by day *t*; missing replies count as `unknown`.
* **Configs:** tune on config A; test once on config B with different parameter ranges and a seed set fixed and committed in advance.
* **Demo school:** built from the simulator on the Lyon graph with a fixed seed, with curated "hero" replies injected so the demo narrative stays deterministic.

### Labeled Text
* **Synthetic replies:** multilingual family replies with typos, terse phrasing, and label noise, each tagged with a ground-truth barrier label. Generate them with a **different model family** than the classifier; report the gap between synthetic and hand-set scores as generator bias.
* **Hand-labeled test set:** 60–100 replies written and labeled by the team (mixed languages, terse and ambiguous), double-annotated with Cohen's κ reported. Written labeling rule for `unknown`.
* **Course Pacing Guide (synthetic):** one guide per grade mapping school dates to topics and a source line, as in `backend/app/data/seed.py`.
* **Data rules:** contact graphs are public; replies, logistics layers, and pacing guides are synthetic. Confirm with Arlequin that this satisfies Track 2's "public data only" rule before relying on it for that track.

### Core Data Entities (ML & Graph Scope)
* **Student:** `anonymous_id`, `grade`, `class_sections`, `bus_route`, `household_id`, `neighborhood`
* **Message:** `message_id`, `absence_id`, `direction`, `language`, `text_or_transcript`, `received_at`
* **Reason:** `absence_id`, `barrier_label`, `confidence_score`, `evidence_quote`, `source_message_id`
* **Cluster:** `id` (deterministic), `layers_involved`, `zone`, `member_ids`, `likely_cause`, `observed`, `expected`, `p_value`, `start_date`, `detected_on`, `weekly_revenue_at_risk`, `evidence_message_ids`
* **Correction:** `absence_id`, `old_label`, `new_label`, `corrected_by`, `timestamp`

---

## 3. AI & Graph Pipeline Specifications

### 3.1 Multilingual Voice & Reason Classification (F1 / F2)
* **Speech-to-Text:** send voice notes to **Gemini's native audio input**, which transcribes and classifies in one call. Whisper is a fallback only. Low-confidence or unsupported languages (e.g., Mixteco, Triqui) go to a human interpreter.
* **Barrier Classification:** **Gemini** maps the text/transcript to one label from the taxonomy in PRD 1 F1.
* **Structured Output:** constrained JSON `{ "label": enum, "confidence": float, "evidence_quote": string }`. The label must match the enum; `evidence_quote` must be an exact substring of the input or it is dropped.
* **Confidence:** self-reported confidence is poorly calibrated, so use agreement across 5 samples at T = 0.7. Choose the auto-accept threshold τ on a dev split so auto-accepted labels reach ≥ 0.95 precision; everything else goes to the clerk queue. Report the share sent to clerks, a risk–coverage curve, and ECE.
* **Latency:** 6 s timeout, keyword fallback, disk cache of responses (also used by demo mode).
* **Learning From Corrections (F8):** staff-confirmed corrections are stored per school; the most similar past corrections are added to the prompt as examples. Corrections used as examples must come from a split that does not overlap the test set.

### 3.2 Root-Cause Cluster Detection (F3 — P0)
* **Zones Z:** every group in each logistics layer (class section, route, neighborhood), plus k-nearest contact ego-nets for every student with k ∈ {5, 10, 20} ranked by contact weight. Households are collapsed to one unit (siblings out together is expected, not a cluster).
* **Windows W:** the last w ∈ {1, 2, 3} school days, run prospectively each day with an `as_of` date (needed for replay and lead time).
* **Statistic:** for each barrier *b*, a Kulldorff space-time permutation scan. With `c` = b-absences in Z×W, `C_Z`, `C_W` the zone and window totals, `C` the total over a baseline of ≥ 40 simulated days, and `μ = C_Z · C_W / C` (which conditions out school-wide waves):
  `LLR = c · ln(c/μ) + (C − c) · ln((C − c)/(C − μ))` when `c > μ`, else 0.
* **Null & multiple testing:** shuffle the dates of b-absences across students 999 times and take the max LLR over all (Z, W) in each (layer, barrier) family, giving a family-wise p-value. Apply Benjamini–Hochberg at q = 0.10 across families. Vectorized NumPy; runs on a laptop.
* **Cause Attribution:**
  * `illness` over contact ego-nets on consecutive days $\rightarrow$ **contagion-like illness cluster** (contact edges are used only here).
  * `transportation` concentrated on one route $\rightarrow$ **route failure (school-caused)**.
* **Output:** rank clusters by p-value then size; attach cited source messages and weekly revenue at risk.
* **Revenue at Risk:** `weekly_revenue_at_risk = (absence days by cluster members this school week) × daily_rate`, default $65, editable. Matches `backend/app/services/revenue.py`.
* **Acceptance:** the demo surfaces at least one illness cluster and one route cluster, each backed by cited family messages and a p-value.
* **Fallback:** if the scan is not ready by the cut-off, use rule-based clusters, labeled as such in the UI.

### 3.3 Grounded Plan Generation & Verification (F4 / F5)
* **Structured Generation:** generate action plans and catch-up plans strictly from structured inputs (cluster metadata, cause, pacing guide, missed dates) rather than open-ended free text.
* **Claim Verification (F5 — P1):** check every generated plan statement against the pacing guide and source materials with an NLI/claim-verification step; flag unsupported claims.

### 3.4 Counselor Assistant (F7 — P1, LangChain Deep Agents track)
* Built as a **Deep Agents** agent with **read-only** tools over the existing API: cluster lookup, message search (keyword/SQL first; pgvector on TigerData if time allows), and a safety check sub-agent. It writes a cited case report to its file system. It has no send tool.
* Constrain answers to retrieved context with explicit message citations.
* **Retrieval benchmark:** 40–60 templated queries (route, cause, week, cross-lingual) with relevant sets R(q) taken from simulator ground truth. `Recall@5 = |top5 ∩ R| / min(5, |R|)`. Compare BM25, dense, and dense + metadata filter. Report citation precision (cited IDs in R).

### 3.5 Safety Triage Classifier (F2 — P0)
* A dedicated safety check runs on every inbound message in every language to detect abuse, neglect, supervision concerns, bullying, or threats of self-harm/violence.
* **The keyword check always runs, and Gemini may add a safety flag but never remove one.** Only a counselor can downgrade a safety label.
* **Fail closed:** if the model errors or the language is unsupported, a human reviews the message.
* Keyword lists cover English, Spanish and Vietnamese, including indirect phrasings ("doesn't want to live", "no quiere vivir"). Normalize curly apostrophes and match whole words.
* A flag stops all automated messages to the student's household and goes to the counselor (process in PRD 3 §3).

---

## 4. Evaluation Framework & Target Metrics

| Metric | Target | Measurement Methodology |
| :--- | :--- | :--- |
| **Barrier Classification** | Macro-F1 $\ge 0.80$ on the hand-labeled set | Macro-F1 with bootstrap 95% CI on synthetic and hand-labeled sets; per-language accuracy; keyword stub as baseline; κ as ceiling. |
| **Auto-accept precision** | $\ge 0.95$ at chosen τ | Risk–coverage curve on dev; share routed to clerks. |
| **Cluster Detection** | Higher event recall than both baselines at equal false-alarm rate | Ablation (below) on held-out config B and the held-out graph. |
| **Early Warning** | % of events detected by onset + 2 days, vs. baseline | Paired per-event delay; median and IQR. |
| **Assistant Retrieval (F7)** | Recall@5 $\ge 0.85$ | Templated benchmark (§3.4). |
| **Safety Escalation** | 0 misses on N frozen seeded messages | Report the 95% lower bound on recall (1 − 3/N) and the false-alarm rate. Separate dev and test suites, frozen before tuning. |

### Ablation Study Design (Required for Demo & Arlequin Track 2)
* **Definitions:** a planted event is `e = (barrier b_e, absentees M_e, onset t_e, duration D_e)`. An alarm is `a = (zone Z_a, barrier b_a, day t_a)`. An alarm is a true positive if `b_a = b_e`, `|Z_a ∩ M_e| / |absent in Z_a| ≥ 0.5`, and `t_a ∈ [t_e, t_e + D_e + 2]`. Alarms with Jaccard > 0.5 across days merge into one.
* **Event recall** = events with ≥ 1 true positive ÷ planted events.
* **Equal false-alarm rate:** calibrate each detector's threshold on null-only runs to 1 false alarm per 20 school days, then compare recall. Also plot recall against false alarms per month. (Fixed precision moves with event prevalence, so it is not used.)
* **Lead time:** `d = t_first_TP − t_e`; `lead = d_baseline − d_ours`, paired per event. Raw-count baseline is EARS-C1: school-wide daily count > mean + 2σ over the last 7 days. Report "% detected by onset + 2 days" because some events are never caught by the baseline.
* **Three detectors, same scan, different inputs:**
  1. **Content-only:** the same scan with Z = whole school or grade (labels, no network).
  2. **Graph-only:** the same scan on all-cause absences (network, no labels), with the cause guessed from zone type.
  3. **Combined (ours):** labels + all zones.
* **Also report:** attribution accuracy (confusion matrix of illness vs route vs other), a label-noise sweep (classifier accuracy 0.5 → 1.0) with an oracle-label row, and the ego-net vs class-section comparison from §1.

### Mitigating Evaluation Circularity
* Simulator mechanisms the detector does not encode (§2), including an untargeted event type and school-wide waves that must not alarm.
* Reply delay and missingness, so the combined detector gets no same-day advantage.
* Tune on config A and the Marseille graph; test once on config B and the Lyon graph with a committed seed set.
* A label-noise curve and a hand-labeled text set; the contact graph itself is real.
