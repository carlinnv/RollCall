# TODO: what is left from the PRDs

Snapshot from Nov 14, 2026, after the Photon channel PR (#3) merged. Nothing else has landed on `main` since the original stubs. Git-ignored.

Owners follow PRD 3 §5: **A** Messaging & Photon (me), **B** ML & Graph, **C** Product & Frontend, **D** Data & Simulator.

## Gates

**Gate 1:** a phone text shows a label and evidence on the dashboard within 10 seconds, and the same flow works with Photon off.

- [x] Message log with latency (A)
- [x] Live dashboard refresh (C)
- [x] Gemini classifier (B)
- [x] Photon-off path (the "Simulate family reply" box)

**Gate 2:** the combined detector beats the content-only and graph-only baselines on held-out runs.

- [ ] Simulator (D)
- [ ] Scan statistic and ablation (B)

## A. Messaging & Photon (mine)

- [x] Message log: one record per inbound and outbound text, many per absence, with real timestamps and `latency_ms`. Needed for Gate 1.
- [x] Plain-language opt-outs in English, Spanish and Vietnamese ("no más mensajes"). Today only exact keywords work.
- [x] Closing an escalation, so household suppression can be lifted.
- [x] Counselor acknowledgement and the 15-minute timer on safety flags. (Acknowledge button and an overdue flag after 15 minutes.)
- [ ] Paging the on-call counselor when a case opens, and the after-hours path (on-call administrator, then principal). The overdue flag only shows on the inbox.
- [ ] Fixed crisis-resource message (988/911) for self-harm or imminent danger. Never for abuse disclosures.
- [ ] Quiet hours, 8am–8pm; queue messages outside them.
- [ ] Voice notes through Gemini audio. Blocked on Photon Beta; needs an in-app upload. See `CARLINNOTES.md`.
- [ ] SMS fallback. Architecture only for now. See `CARLINNOTES.md`.
- [ ] Demo mode with a replayable message fixture. "Never cut" in the PRD.
- [x] Update `.env.example`: it still says "ask Carlin for these", which contradicts `PHOTON_SETUP.md`.

- [ ] Persist case reports (and the rest of the data). Saved case reports live in memory and are lost on restart or reset.

## B. ML & Graph

- [x] Gemini classifier replacing the keyword stub: structured JSON output, evidence quote must be an exact substring, 6-second timeout with keyword fallback, response cache.
- [ ] Taxonomy: the code has 8 barrier labels; PRD 1 lists 14 (adds `school-caused`, `mental health or school avoidance`, `late arrival`, `community safety concern`, `religious or cultural observance`, and splits housing and food).
- [x] Safety classifier: Vietnamese terms, indirect phrasings, whole-word matching; the model may add a flag but never clear one; fail closed. (Vietnamese terms still need a native speaker's review.)
- [ ] Confidence from agreement across 5 samples, with an auto-accept threshold.
- [ ] Cluster detection: permutation scan statistic with p-values, replacing the rule-based stub.
- [ ] Three detectors and the held-out ablation (`eval/ablation.json`).
- [ ] Learning from corrections (F8).
- [ ] Counselor assistant (F7), time-boxed to 3 hours after Gate 2.

## C. Product & Frontend

- [ ] Morning briefing (F9) as the clerk's home screen.
- [ ] Agent actions (F8): drafted recode flags, transport and nurse notices, with Approve / Edit / Dismiss.
- [x] Live refresh (polling or server-sent events). See `CARLINNOTES.md`.
- [ ] Check-in button on the case page (the endpoint exists).
- [ ] Label-correction UI.
- [ ] Role switcher and role-based views: clerks do not see safety disclosure text; housing cases go to the liaison only.
- [ ] Approval by owning role for plans (clerk, counselor, liaison, nurse).
- [ ] Revenue dashboard: hide cells under 11 students; "pre-hold-harmless estimate" label.
- [ ] Audit tab.
- [ ] Aggregate graph view and ablation table (Arlequin track).
- [ ] Suggested SIS code per absence.
- [ ] "Synthetic data — fictional school" banner.
- [ ] Clerk edits to a plan message should reach families in every language, not only the plan's language.

## D. Data & Simulator

- [ ] Simulator on the SocioPatterns Lyon and Marseille graphs, with fixed seeds, planted events, reply delay and ground truth.
- [ ] Demo school built from the simulator, with curated "hero" replies.
- [ ] Synthetic multilingual replies, generated with a different model family than the classifier.
- [ ] Hand-labeled test set of 60–100 replies, double-annotated.
- [ ] TigerData sidecar: hypertable, continuous aggregate, timeline chart. Time-boxed to 3 hours after Gate 1.
- [ ] Scaling benchmark chart.

## Optional tasks

- [ ] Gemini opt-out flag: have the Gemini classifier also return whether a message is asking us to stop messaging, to catch phrasings the keyword and phrase patterns miss ("enough already", "ya basta con esto"). The patterns in `outreach.py` keep running, and the model may add an opt-out but never remove one. Depends on the Gemini classifier (B).

## Whole team

- [ ] Freeze the data contracts (Message, Cluster v2, AgentAction, Correction, Escalation).
- [ ] Deploy to Vultr with HTTPS and a domain.
- [ ] Connect ChatOverflow.
- [ ] Hash-chained audit log.
- [x] Replace names with placeholders before any model call. (Student, guardian and teacher names; done in `classifier.py`.)
- [ ] Disable `/api/reset` in the public build.
- [ ] Decide on renaming the product to "Here".
- [ ] Three timed demo rehearsals and a backup video.
- [ ] Devpost, README, DEV post.

## Questions to settle

- [ ] Arlequin: does a real public contact graph with synthetic replies satisfy Track 2's "public data only" rule?
- [ ] Peri: is a defined persona plus CSV workflow enough, or do judges expect a named pilot district?
- [ ] Photon: will they comp a dedicated line, or let credits apply? (Cold outbound; see `CARLINNOTES.md`.)
- [ ] Photon: is there an SMS number that skips 10DLC for hackathon accounts?
- [ ] Verify before the pitch: CA 2024–25 chronic absence rate, Attendance Recovery rules, Utah dataset license.
- [ ] Do teammates have unpushed work? Nothing from B, C or D is on the remote yet.
- [ ] Team: should outbound messages tell families how to opt out and back in? Today nothing mentions STOP or START, so families can only find them by guessing. Things to decide:
  - Which messages carry it: only the first check-in, or every message.
  - The wording, in each language ("Reply STOP to stop these messages" / "Responda ALTO para no recibir más mensajes"). Translations need human review.
  - Whether to confirm an opt-out. Today the agent goes silent, so a family cannot tell it worked. A one-time confirmation is the carrier norm for SMS but it means one more message after "stop".
  - Whether to confirm START.
  - It makes the supportive first check-in longer and more formal; weigh that against the PRD's consent and opt-out rules (PRD 3 §3) and the 10DLC requirements if SMS is added.

## Already answered (tell the team)

- Photon does not need a macOS host.
- Photon Beta has no voice notes and no automatic SMS fallback.
- Polling needs no public URL, so Photon does not depend on the Vultr deploy.
- PRD 3 says Python 3.14 is unsupported by the pinned pydantic. That pin was raised and tests pass on 3.14; the Photon client needs 3.11 or later.
