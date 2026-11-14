# PRD 1: Product, UX & Business Impact — Why Absent

**Date:** Nov 14, 2026 | **Owner:** @Mayukha  
**Focus:** Problem definition, user personas, core product flows, dashboards, plans, and demo narrative.

> **Open decision (team):** rename to **"Here"** (the roll-call answer), tagline *"From 'unexcused' to 'the bus never came.'"* Reviewers found "Why Absent" reads as an accusation, which clashes with the families-as-partners stance. "RollCall" is generic.

---

## 1. Overview & Problem Statement
* **Overview:** Why Absent finds out why students miss school, traces clusters of absences to a shared root cause, and turns each cause into an action the school can take. The estimated attendance-funding impact is shown only in aggregate, never per student.
* **Problem:** A district attendance clerk knows that a student was absent, usually recorded simply as excused or unexcused, but rarely knows the underlying barrier. Outreach is generic, slow, and often aimed at the wrong fix. Worse, absences the district caused (a bus that never came) can sit coded as *unexcused* and push families toward truancy letters.
* **Primary user:** the attendance clerk at a mid-size district in a state that funds schools by attendance (California is the reference state). Buyer: the business manager. Partners: counselor, nurse, McKinney-Vento liaison.

### Market Context & Economic Opportunity
* **Revenue Link:** In attendance-funded states like California, ADA funding is lost for every absence, excused or not. California's Attendance Recovery program lets districts recover funding for days made up in **supervised instruction outside the regular school day** (up to 10 days per student per year; certificated supervision; verify current rules with CDE). Self-paced catch-up work alone does not recover funding.
* **Hold-harmless caveat:** LCFF funds on the greater of current, prior, or three-year-average ADA, so a single week's loss is often lagged or partly protected. All dollar figures here are pre-hold-harmless estimates.
* **Scale:** California's chronic absenteeism (missing ≥10% of enrolled days) peaked near 30% in 2021–22 against ~12% pre-pandemic (2018–19). The 2024–25 figure (~19%, ~1.1M students) needs verifying against CDE DataQuest before it goes in the pitch.
* **Dollar Value:** Vendor estimates (unverified) put the daily rate at **$35–$100**; the product defaults to $65 and makes it editable.
* **Messaging Efficacy:** In a national IES trial of ~26,000 elementary students, adaptive texting lowered the expected 20.5% chronic absence rate by **2.4 to 3.6 percentage points** at a cost of **$6.90 to $8.53 per student per year** (no direct reading or math score gains in the study year).

#### Rough Economics per 1,000 Students (to recompute)
* **Method:** `benefit = absence-days avoided × daily rate + days made up in Attendance Recovery sessions × daily rate`, net of program cost.
* The earlier table counted 18 recovered days for every student who stopped being chronically absent. A student who goes from 19 absences to 17 only avoids 2 days, so those figures overstated the benefit several times and have been removed. Recompute from the simulator's absence-days before quoting a number.

* **Competitive Gap:** Incumbents (e.g., SchoolStatus, ParentSquare) automate attendance messaging, but no tool captures structured barriers across families and links them across students to uncover shared systemic causes. *"They send messages; we read the replies and connect them across families. One fix helps five kids."*

---

## 2. Users & Personas

| Persona | Job to be Done | Main Surface |
| :--- | :--- | :--- |
| **Attendance Clerk** *(Primary User — name them in the pitch)* | Learn why each student was out, post the right SIS code, and act without calling every family. | Morning briefing, case queue, action approvals |
| **Business Manager** *(Buyer)* | See aggregate funding impact and which root cause costs the most. | Revenue dashboard (aggregates only) |
| **Counselor / Site Admin** | Handle safety, supervision, mental-health and engagement cases; own mandated-reporting follow-up. | Escalation inbox |
| **School Nurse** | Own illness guidance and outbreak response. | Illness cluster notices |
| **McKinney-Vento Liaison** | Handle housing-insecurity cases confidentially. | Liaison inbox |
| **Teacher** | Know what an absent student missed and reteach it. | Missed-topics list |
| **Family or Caregiver** | Report a reason quickly in their own language and get help with the barrier. | Text or voice conversation |

* **Design Stance:** Families are treated as partners, not violators. Outreach asks what is getting in the way and offers support; nothing in the product frames an absence as punishment. When the school caused the absence, the message says so.

---

## 3. Product Goals & Non-Goals

### Goals
1. Capture a structured barrier reason for each absence with a single family reply (or from an existing call-in reason).
2. Detect clusters of shared causes by combining reason text with the school's recorded networks (routes, sections, contacts).
3. Turn each root cause into actions the agent drafts and staff approve, and each student with 2+ missed days into a catch-up plan.
4. Show aggregate funding at risk and recovered per root cause.
5. Adapt: clerk corrections improve future classifications.

### Non-Goals
* Replacing the Student Information System (SIS) or building live SIS integrations this weekend (CSV in, suggested codes out).
* Using real student data (synthetic or public anonymized data only).
* Inferring social friendships between students.
* Recording immigration status or details of immigration-related fears.
* Automating truancy referrals, discipline, or punitive decisions.
* Using dollar value to prioritize which students receive help (**risk of harm sets priority; revenue justifies the program**).

---

## 4. Core Product Features (User-Facing)

### F1. Absence Tracking & Cause Capture (P0)
* Ingest a daily absence CSV export from an SIS (including tardies and partial days) and maintain each student's absence history. If a call-in or note already gives a reason, record it and skip outreach.
* Store one structured barrier per absence from the taxonomy: `illness`, `transportation`, `housing insecurity`, `food insecurity`, `caregiving or family obligation`, `mental health or school avoidance`, `safety or bullying`, `disengagement or academic struggle`, `appointment or travel`, `religious or cultural observance`, `late arrival`, `school-caused` (e.g., bus did not run), `community safety concern` (stores no free-text details), or `unknown`.
* Display original reply text and voice transcripts alongside the classified label for clerk verification, and suggest an SIS attendance code for the clerk to post.
* **Acceptance:** Every absence in the demo file shows status (`reason captured`, `awaiting reply`, or `unknown`), and the clerk can inspect the source message.

### F4. School Action Plan for Families (P0)
* Generate a structured plan per cluster or individual case containing: action the school will take, internal owner, target date, and what is requested of the family.
* Draft a plain-language outbound message in the family's preferred language (greeting by title and surname, correct gendered grammar).
* **Approval by role:** the clerk approves transportation, appointment and unknown cases; the counselor approves disengagement, caregiving and mental-health cases; the McKinney-Vento liaison approves housing cases; the nurse writes illness guidance (the clerk never sends medical advice).
* **Acceptance:** From a detected cluster, the right approver approves and sends an action plan in under 1 minute.

### F5. Catch-Up Plan (P0 Generation / P1 Verification)
* For students who missed **2+ days**: a teacher-facing missed-topics list mapped from the pacing guide, one small-group reteach suggestion, a paper take-home in the home language, and an offer of an Attendance Recovery session.
* Adapt to the cause (lighter load after illness; short check-ins for disengagement). No "self-paced video" plans for K–5.
* *(P1)* Verify every claim in the plan against the pacing guide and source notes, flagging unsupported statements.
* **Acceptance:** The plan includes only topics present in the pacing guide and scales with missed days. Only supervised Attendance Recovery sessions count toward recovered funding.

### F6. Revenue Dashboard (P0)
* Display aggregate funding at risk this week, funding recovered to date, and top root causes ranked by dollar impact. Suppress any cell with fewer than 11 students.
* Provide editable assumptions for daily funding rate and state rules (defaulting to California), with a "pre-hold-harmless estimate" label.
* **Acceptance:** Updating the daily rate recalculates all dollar figures across the dashboard.

### F8. Agent Actions & Feedback Loop (P0)
* When a cluster or case is detected, the agent drafts the follow-up actions itself, not only a plan:
  * **Route cluster (school-caused)** $\rightarrow$ flag the affected absences for the clerk to recode per district policy, hold any truancy letters, notify the principal, and draft a family message: *"Route 7 did not run Tuesday and Wednesday. We're sorry. This was our problem, not yours."* A transport notice may go out with **route, dates and rider count only** (no names, replies or barriers).
  * **Illness cluster** $\rightarrow$ notice to the school nurse with class section, dates, count and symptom category; names are opened in the dashboard, not sent in the message.
  * **Unclear reply** $\rightarrow$ one short follow-up question to the family in their language.
  * **Supervision concern** (e.g., young children home alone caring for an infant) $\rightarrow$ counselor/admin review with a mandated-reporter prompt, never a routine childcare message.
  * **Other individual case** $\rightarrow$ task for the owning role (F4) with the evidence messages attached, inside the app only.
* Every action is a one-click **Approve / Edit / Dismiss** for the owning role. Safety cases never get an automated action.
* **Learning from corrections:** when staff correct a barrier label, the correction is stored and the most similar past corrections are used as examples for future classifications at that school.
* **Acceptance:** From one detected route cluster, the agent drafts the recode flags and the family messages, and the clerk sends both in under 1 minute. After one label correction, a similar reply is classified correctly.

### F9. Morning Briefing (P0)
* The clerk's home screen each morning: new clusters since yesterday, replies needing review, drafted actions waiting for approval, absences needing an SIS code, and safety cases still open (status only, never the disclosure text).
* Example: *"3 new clusters · 12 replies need review · Route 7 did not run (5 riders, 2 days) · recode flags and family message drafted."*
* **Acceptance:** Every item on the briefing links to the case, cluster, or drafted action behind it.

### Outreach Rules
* The first check-in asks one open, supportive question and says who reads replies. It does not state an absence count.
* Count-based messages only at chronic-absence thresholds (CA definition: ≥10% of enrolled days, ≥31 days enrolled), tiered at 5–9%, 10–19% and 20%+.
* Coordinate with the district's existing robocall so a family gets one message, not three.

---

## 5. End-to-End User Flow & 3-Minute Demo Script

### User Flow
1. **Absence Logged** $\rightarrow$ If no reason is on file, the system sends a supportive check-in.
2. **Family Replies** (text or voice note in any language).
3. **Safety Gate:** If safety-critical $\rightarrow$ **counselor takes over**; all automated messages to that household stop until the counselor closes the case.
4. **Classification & Pricing:** If non-safety $\rightarrow$ barrier labeled $\rightarrow$ suggested SIS code $\rightarrow$ aggregate dashboard updates.
5. **Routing:** If part of a cluster $\rightarrow$ cluster-level actions drafted; otherwise $\rightarrow$ case-level actions drafted for the owning role.
6. **Human Approval & Recovery:** Owner approves and sends $\rightarrow$ catch-up plan for students with 2+ missed days.
7. **Feedback:** Staff corrections to labels feed back into future classifications.

### 3-Minute Demo Script
* **0:00** — *"Monday, 9am. Maria is the only attendance clerk for 1,500 students. Five kids on Route 7 are marked unexcused. Their parents each told the school the same thing, in three languages: the bus never came. Nobody connected it."*
* **0:20** — Morning briefing. Route 7 shows 2 of 3 matching replies, not a cluster yet.
* **0:35** — A live Spanish voice reply arrives; label and evidence quote appear, and **the Route 7 cluster lights up** (the "ah-ha").
* **1:05** — Why both layers matter: the text-only detector misses it and the graph-only detector calls it illness. Flash the benchmark table.
* **1:35** — The agent has drafted recode flags and an apology message to families. Maria approves both in one click.
* **2:05** — Maria corrects one wrong label, and the next similar reply comes back right.
* **2:35** — *"Five kids back on a bus by Wednesday, with no truancy letters. It also recovers about $X a week in funding, which is how the district pays for it."* Close on kids, not dollars.
* **Backup:** deterministic demo mode with cached model responses and a recorded video.
