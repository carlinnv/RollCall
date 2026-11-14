# PRD 1: Product, UX & Business Impact — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** Problem definition, user personas, core product flows, dashboards, plans, and demo narrative.

---

## 1. Overview & Problem Statement
* **Overview:** Why Absent finds out why students miss school, traces clusters of absences to a root cause, and turns each cause into an action the school can take, priced in dollars per week.
* **Problem:** A district attendance clerk knows that a student was absent, usually recorded simply as excused or unexcused, but rarely knows the underlying barrier. Outreach is generic, slow, and often aimed at the wrong fix.
* **Customer:** The attendance clerk, business manager, and counselor at a mid-size district in a state that funds schools by attendance (California is the reference state).

### Market Context & Economic Opportunity
* **Revenue Link:** In attendance-funded states like California, districts lose funding whenever a student misses school for any reason. California’s Attendance Recovery program allows districts to offer make-up instruction to recover funding tied to those days.
* **Scale:** California's chronic absenteeism rate was 19% in 2024–25 (~1.1 million students), down from a 30% peak in 2021–22, but still well above the 12% pre-pandemic baseline (2018–19).
* **Dollar Value:** Vendor estimates (unverified) place the cost of a chronically absent student at **$630 to $1,800 per year** ($35–$100 daily rate $\times$ 18 days).
* **Messaging Efficacy:** In a national IES trial of ~26,000 elementary students, adaptive texting lowered the expected 20.5% chronic absence rate by **2.4 to 3.6 percentage points** at a cost of **$6.90 to $8.53 per student per year** (though it showed no direct reading or math score gains in the study year).

#### Rough Economics per 1,000 Students
*(Low benefit paired with high cost; high benefit paired with low cost)*

| Case | Fewer Chronically Absent Students | Revenue Recovered | Program Cost | Net Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **Conservative** | 24 | $15,120 | $8,530 | **$6,590** |
| **Optimistic** | 36 | $64,800 | $6,900 | **$57,900** |

* **Competitive Gap:** Incumbents (e.g., SchoolStatus) automate attendance messaging, but no tool captures structured barriers across families and links them across students to uncover shared systemic causes.

---

## 2. Users & Personas

| Persona | Job to be Done | Main Surface |
| :--- | :--- | :--- |
| **Attendance Clerk** *(Primary User)* | Learn why each student was out and decide what to do without calling every family. | Case queue, action-plan sender |
| **Business Manager** *(Buyer)* | See revenue at risk, dollars recovered, and which root cause costs the most. | Revenue dashboard |
| **Counselor or Nurse** | Handle safety, health, and engagement cases escalated by the clerk. | Escalation inbox, root-cause clusters |
| **Teacher** | Help an absent student catch up on missed instruction. | Learning plan |
| **Family or Caregiver** | Report an absence reason quickly in their own language and get help with the barrier. | Text or voice conversation |
| **Student** | Catch up on missed material at their own pace. | Personalized learning plan |

* **Design Stance:** Families are treated as partners, not violators. Outreach asks what is getting in the way and offers support; nothing in the product frames an absence as punishment.

---

## 3. Product Goals & Non-Goals

### Goals
1. Capture a structured barrier reason for each absence with a single family reply.
2. Detect clusters of shared causes by combining reason text with the school contact network.
3. Convert each root cause into a school action plan and each absent student into a catch-up learning plan.
4. Quantify revenue at risk and revenue recovered per root cause.

### Non-Goals
* Replacing the Student Information System (SIS) or building live SIS integrations this weekend.
* Using real student data (synthetic or public anonymized data only).
* Inferring social friendships between students.
* Automating truancy referrals, discipline, or punitive decisions.
* Using dollar value to prioritize which students receive help (**risk of harm sets priority; revenue justifies the program**).

---

## 4. Core Product Features (User-Facing)

### F1. Absence Tracking & Cause Capture (P0)
* Ingest a daily absence CSV export from an SIS and maintain each student's absence history.
* Store one structured barrier per absence from the taxonomy: `illness`, `transportation`, `housing or food insecurity`, `caregiving or family obligation`, `safety or bullying`, `disengagement or academic struggle`, `appointment or travel`, or `unknown`.
* Display original reply text and voice transcripts alongside the classified label for clerk verification.
* **Acceptance:** Every absence in the demo file shows status (`reason captured`, `awaiting reply`, or `unknown`), and the clerk can inspect the source message.

### F4. School Action Plan for Families (P0)
* Generate a structured plan per cluster or individual case containing: action the school will take, internal owner, target date, and what is requested of the family.
* Draft a plain-language outbound message in the family's preferred language.
* Require human clerk review and approval before sending; track plan status.
* **Acceptance:** From a detected cluster, a clerk approves and sends an action plan in under 1 minute.

### F5. Personalized Learning Plan (P0 Generation / P1 Verification)
* Map each absent student's missed dates to the course pacing guide to identify missed topics.
* Generate a sequenced catch-up plan with time estimates and teacher check-ins, adapted to the absence cause (e.g., lighter workload following an illness).
* *(P1)* Verify every claim in the learning plan against the pacing guide and source notes, flagging unsupported statements.
* **Acceptance:** The generated plan includes only topics present in the pacing guide and scales appropriately to the number of missed days.

### F6. Revenue Dashboard (P0)
* Display total revenue at risk this week, revenue recovered to date, and top root causes ranked by dollar impact.
* Provide editable assumptions for daily funding rate and state rules (defaulting to California).
* **Acceptance:** Updating the daily rate dynamically recalculates all dollar figures across the dashboard.

### F8. Agent Actions & Feedback Loop (P0 Actions / P1 Learning)
* When a cluster or case is detected, the agent drafts the follow-up actions itself, not only a plan:
  * **Route cluster** $\rightarrow$ alert to the transportation office with affected route, dates, and rider count.
  * **Illness cluster** $\rightarrow$ notice to the school nurse with affected class sections and dates.
  * **Unclear reply** $\rightarrow$ one short follow-up question to the family in their language.
  * **Individual case** $\rightarrow$ counselor or family-resource task with the evidence messages attached.
* Every action is a one-click **Approve / Edit / Dismiss** for the clerk. Safety cases never get an automated action; they go to a counselor.
* *(P1)* When the clerk corrects a barrier label, the correction is stored and used as an example for future classifications at that school, and repeated corrections adjust that school's review threshold.
* **Acceptance:** From one detected route cluster, the agent drafts the transport alert and the family messages, and the clerk sends both in under 1 minute. After one label correction, a similar reply is classified correctly.

### F9. Morning Briefing (P0)
* The clerk's home screen each morning: new clusters since yesterday, replies needing review, drafted actions waiting for approval, and safety cases still open.
* Example: *"3 new clusters · 12 replies need review · Route 7 likely broken down (5 riders, 2 days) · transport alert drafted."*
* **Acceptance:** Every item on the briefing links to the case, cluster, or drafted action behind it.

---

## 5. End-to-End User Flow & 3-Minute Demo Script

### User Flow
1. **Absence Logged** $\rightarrow$ System sends supportive check-in message with absence count.
2. **Family Replies** (text or voice note in any language).
3. **Safety Gate:** If safety-critical $\rightarrow$ **Counselor takes over immediately** (no automated reply).
4. **Classification & Pricing:** If non-safety $\rightarrow$ Barrier labeled $\rightarrow$ Revenue dashboard updates dollars at risk.
5. **Routing:** If part of a cluster $\rightarrow$ Cluster-level action plan drafted; otherwise $\rightarrow$ Individual case-level plan drafted.
6. **Agent Actions:** Agent drafts the follow-up actions for the cluster or case (transport alert, nurse notice, family follow-up, counselor task).
7. **Human Approval & Recovery:** Clerk approves and sends the actions and family action plan $\rightarrow$ System generates personalized student learning plan.
8. **Feedback:** Clerk corrections to labels feed back into future classifications.

### 3-Minute Demo Script
* **0:00** — Clerk opens the morning briefing, then the queue showing dozens of absences with no recorded reasons.
* **0:20** — Live iMessage demo: family replies with a Spanish voice note; within 10 seconds, transcript, barrier label, and evidence quote appear.
* **1:00** — Cluster view highlights one bus-route cluster and one illness cluster, showing ablation results and weekly dollars at risk.
* **1:50** — Clerk approves the agent's drafted transport alert and school action plan; plain-language Spanish response is dispatched to affected families.
* **2:30** — Personalized catch-up learning plan appears for an absent student, adapted to their absence cause.
* **2:50** — Close on total dollars recovered and target district pilot metrics.