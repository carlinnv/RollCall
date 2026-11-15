# Features on `carlin/photon_setup`

What this branch adds: the Photon 2.0 (Beta) iMessage channel for PRD feature F2. Each section lists the files a feature uses and the rules it follows. Last updated Nov 14, 2026.

To set the channel up on your machine, see [PHOTON_SETUP.md](PHOTON_SETUP.md).

All paths are relative to the repo root. Tests for every feature are in `backend/tests/test_api.py` and run with Photon off.

---

## 1. Photon connection

Connects the backend to Photon 2.0 with its Python client.

**Files**

- `backend/requirements.txt`: adds `photonhq-api==0.5.0`; `pydantic` pin raised to `>=2.13.5,<2.15`
- `.env.example`: the variables each developer needs
- `backend/app/main.py`: loads `.env` and starts the channel on server startup
- `backend/scripts/photon_echo.py`: one-shot check that waits for a text from your phone and echoes it back

**Rules**

- The channel starts only when `PHOTON_API_KEY`, `PHOTON_PROJECT_ID`, `PHOTON_AGENT_HANDLE` and `PHOTON_TEST_SENDER` are all set. Otherwise it stays off and the app works as before.
- Tests never start the channel.
- Requires Python 3.11 or later.

---

## 2. Inbound messages

A text from a family is classified and stored through the same path as the "Simulate family reply" box.

**Files**

- `backend/app/services/photon_channel.py`: polling loop and `handle_inbound`
- `backend/app/store.py`: `record_reply` (safety gate, classification, routing; unchanged)

**Rules**

- Photon is polled every 2 seconds.
- A message is processed only if all of these hold:
  - it is an iMessage with status `received`
  - it is from a human to the agent line in `PHOTON_AGENT_HANDLE`
  - it is a direct conversation, not a group chat
  - its content is plain text (images and other content are skipped)
- Each message is processed once, tracked by its Photon message ID.
- Messages that arrived before the server started are not classified.
- Message text is passed to the classifier as data only. It never triggers an action.
- A failed poll is logged and retried on the next cycle; the loop does not stop.

---

## 3. Matching a phone number to a student

Decides which student and which absence an inbound text belongs to.

**Files**

- `backend/app/services/photon_channel.py`: `_links`, `handle_inbound`
- `backend/app/services/outreach.py`: `digits`, `phone_for`, `demo_student_id`

**Rules**

- Numbers are compared by their last 10 digits.
- A number matches the student whose `guardian_phone` it equals.
- The number in `PHOTON_TEST_SENDER` is treated as the guardian of `PHOTON_DEMO_STUDENT_ID` (default `S002`).
- The reply is attached to the student's most recent absence that is awaiting a reply. If none is waiting, it goes to their most recent absence.
- A number that matches no student is not classified. It gets a fixed "call the school office" reply.
- The reply's language is taken from the student's `preferred_language`. It is not detected from the text.

---

## 4. Automatic acknowledgement

A short fixed reply so the family knows the message arrived.

**Files**

- `backend/app/services/photon_channel.py`: `ACK`, `handle_inbound`, `_poll`

**Rules**

- The text is a fixed template in English or Spanish, chosen by the student's `preferred_language`. Other languages get English.
- It is never model-generated and contains no links.
- It must pass the outbound rules in section 6. In particular, no acknowledgement is sent when the household has an open safety escalation.
- It is sent as a reply to the family's message, with idempotency key `rollcall-ack:<message id>`, so a retry cannot send it twice.

---

## 5. Opt-out and opt-in

A family can ask us to stop messaging, with a keyword or in their own words.

**Files**

- `backend/app/services/outreach.py`: `OPT_OUT_WORDS`, `OPT_IN_WORDS`, `_OPT_OUT_PHRASES`, `is_opt_out`, `is_opt_in`, `opted_out`
- `backend/app/services/photon_channel.py`: `handle_inbound`

**Rules**

- Before matching, text is lowercased and stripped of accents and punctuation, so "No más mensajes." and "no mas mensajes" are the same.
- **Keywords.** The whole message is one of:
  - English: `stop`, `stopall`, `unsubscribe`, `cancel`, `end`, `quit`
  - Spanish: `alto`, `parar`, `pare`, `basta`, `baja`
  - Vietnamese: `dừng`, `ngừng`, `hủy`
- **Plain language.** The message contains a request that names messaging, anywhere in the text. Examples that match:
  - English: "please stop texting me", "no more messages", "don't text me", "do not contact us", "take me off this list", "unsubscribe", "leave us alone"
  - Spanish: "no más mensajes", "ya no me escriban", "dejen de mandarme mensajes", "no quiero recibir más mensajes", "quiero darme de baja"
  - Vietnamese: "đừng nhắn tin nữa", "ngừng gửi tin nhắn", "không muốn nhận tin nhắn", "hủy đăng ký"
- Phrases must mention messaging or contact, so ordinary replies do not match: "the bus didn't stop", "she wants to quit school", "we had to cancel the appointment".
- An opt-out takes effect immediately. The number receives no confirmation and no further messages of any kind.
- A message that is only an opt-out is logged but not attached to an absence, so it cannot overwrite a reason already captured.
- A message that opts out and also gives a reason ("no more texts, she has a fever") is honored and classified.
- Texts from an opted-out family are still classified, so a reason or a safety disclosure is never dropped. They get no reply.
- **Opting back in.** A message that is exactly `start` or `unstop` removes the opt-out. No confirmation is sent.
- Opt-outs are held in memory and are lost when the server restarts.
- The Vietnamese phrases have not been reviewed by a Vietnamese speaker.

---

## 6. Outbound rules

The checks every outbound message must pass before it is sent.

**Files**

- `backend/app/services/outreach.py`: `blocked_reason`, `has_link`
- `backend/app/store.py`: `household_suppressed`, `send_to_family`

**Rules**

Checked in this order; the first one that applies blocks the message:

1. **Safety suppression.** No automated message goes to a household while any student in it has a safety escalation the counselor has not closed. Only the counselor contacts that family. See section 11.
2. **Opt-out.** No message goes to a number that has opted out.
3. **No links.** A message is blocked if it contains `http://`, `https://`, `www.`, or a domain ending in `.com`, `.org`, `.net`, `.edu`, `.gov`, `.io`, `.co`, `.ly` or `.us`.
4. **Not empty.** A blank message is blocked.

A blocked message is recorded in the outbox with status `blocked` and the reason. It is not retried.

---

## 7. Outbound sending

Delivers a message that passed the rules.

**Files**

- `backend/app/services/photon_channel.py`: `send_text`, `contacts`, startup catch-up in `_loop`
- `backend/app/store.py`: `sender`, `send_to_family`
- `backend/scripts/photon_send.py`: one-shot check that sends a test message to your phone
- `frontend/src/pages/Plans.tsx`, `frontend/src/api.ts`: status column in the outbound messages table

**Rules**

- A family is reachable only after it has texted the agent line. Photon's shared line cannot send the first message.
- On startup the channel reads the last 3 days of inbound messages to relearn who is reachable. Those messages are not re-classified.
- Every attempt is recorded in the outbox with one status:

  | Status | Meaning |
  | :--- | :--- |
  | `sent` | Photon accepted the message. Delivery is not confirmed. |
  | `queued` | Allowed, but the channel is off or the family has not texted yet. |
  | `blocked` | An outbound rule forbids it. |
  | `failed` | Photon rejected the send. |

- Queued messages are not sent automatically later. The send has to be requested again.
- All messages go over iMessage. There is no SMS fallback.

---

## 8. First check-in message

The supportive first message asking what got in the way.

**Files**

- `backend/app/services/outreach.py`: `CHECK_IN`, `check_in_text`
- `backend/app/store.py`: `check_in_message`, `send_check_in`
- `backend/app/main.py`: `POST /api/absences/{id}/check-in`

**Rules**

- Sent only when the absence has no reply on file. Otherwise the request returns `blocked`.
- Asks one open question, names the attendance clerk as the reader of replies, and offers help.
- States no absence count and does not invite voice notes.
- Written in English or Spanish by the student's `preferred_language`; other languages get English.
- Uses the guardian's name, the student's first name and the date of the absence.
- Idempotency key `rollcall-checkin:<absence id>`: Photon delivers at most one check-in per absence.
- Sent on request through the endpoint. Nothing sends it automatically when an absence is imported.

---

## 9. Approved plan messages

Family messages from a school action plan go out over Photon.

**Files**

- `backend/app/store.py`: `approve_plan`
- `backend/app/services/action_plan.py`: `personalize` (unchanged; `outbox_entry` removed)

**Rules**

- Nothing is sent until a staff member approves the plan.
- One message per recipient family, each checked against the outbound rules on its own. One family being blocked does not stop the others.
- Idempotency key `rollcall-plan:<plan id>:<student id>`.
- The plan is marked `sent` once processed, even if some messages were blocked or queued. The outbox shows the outcome per family.
- Known gap: a clerk's edit to the message reaches only families in the plan's language. Families in another language get the standard template.

---

## 10. Message log and latency

A record of every text between the school and a family, with how long each took.

**Files**

- `backend/app/models.py`: `Message`
- `backend/app/store.py`: `log_message`, `record_reply`, `send_to_family`, `thread`, `latency_summary`, `outbox`
- `backend/app/services/photon_channel.py`: `handle_inbound`, `_poll`, `status`
- `backend/app/main.py`: `GET /api/messages`, `GET /api/messages/latency`, `GET /api/channel`
- `frontend/src/pages/CaseDetail.tsx`, `frontend/src/api.ts`: conversation view with latency

**Rules**

- Every inbound and outbound text is logged, on every channel: iMessage, the "Simulate family reply" box (`simulated`), and the sample data (`seed`).
- An absence can have many messages. The absence itself still shows only its latest reply and label; the log keeps each message with the label it received.
- Inbound texts that never reach the classifier (unknown number, opted out, no absence) are logged with no absence and no label.
- Outbound messages are logged whatever the outcome: `sent`, `queued`, `blocked` or `failed`, with the reason. The outbox on the Plans page is read from this log.
- Timestamps use the real clock in UTC, not the demo date. Sample-data messages keep their scripted times.
- `latency_ms` means:

  | Message | Measured from | Measured to |
  | :--- | :--- | :--- |
  | Inbound iMessage | Photon received the family's text | label stored |
  | Inbound simulated | the request arrived | label stored |
  | Automatic reply | Photon received the family's text | Photon accepted our reply |
  | Check-in or plan message | we started the send | Photon accepted it |
  | Sample data, or outbound not sent | not measured | |

- iMessage latency includes the wait before the next 2-second poll, so it is the delay a family would actually see.
- `GET /api/messages` lists the log, filtered by `absence_id`, `student_id` or `direction`.
- `GET /api/messages/latency` reports how many replies were measured, how many met the 10-second target, and the median and maximum. `?channel=imessage` limits it to real texts.
- `GET /api/channel` returns whether the channel is on and listening, the number of reachable families and opt-outs, the iMessage latency summary, and the last 50 iMessage log entries.
- The case page shows the whole conversation, with "labeled in N s" under each reply.
- The log is in memory and is cleared on restart or reset. It is not the hash-chained audit log the PRD describes.

---

## 11. Safety escalations

What happens after a message is flagged, and how the counselor closes it.

**Files**

- `backend/app/models.py`: `Escalation`
- `backend/app/store.py`: `_route_reply`, `open_escalation`, `escalation_view`, `acknowledge_escalation`, `record_report_decision`, `close_escalation`, `household_suppressed`
- `backend/app/main.py`: `GET /api/escalations`, `POST /api/escalations/{id}/acknowledge`, `/report`, `/close`
- `frontend/src/pages/Escalations.tsx`: the counselor's inbox and case report panel
- `backend/app/services/case_report.py`: builds the internal case report

**Rules**

- A safety-flagged message opens an escalation for that absence, unless one is already open for it.
- While an escalation is not closed:
  - nothing automated is sent to anyone in the student's household
  - the absence keeps its safety label. A calmer follow-up ("she just has a fever") is logged with its own label but does not replace it. Only the counselor can downgrade a safety label.
  - the case stays at the top of the queue and cannot get an automated plan.
- The counselor takes three steps:
  1. **Acknowledge.** The inbox marks a case overdue if it has waited more than 15 minutes without acknowledgement.
  2. **Record the mandated-reporting decision:** report filed, or no report required. The system never files a report; staff record what they did. A case cannot be closed before this step.
  3. **Close**, as either:
     - **concern handled:** the absence keeps its safety label, and still never gets an automated plan
     - **not a safety concern:** the counselor picks the correct label, which replaces the safety label (source "relabeled by the counselor")
- Closing also acknowledges the case if that step was skipped.
- Messages to the household resume when its last open escalation closes. An open case for a sibling keeps the household silent.
- A closed escalation cannot be changed. A new safety message for the same absence opens a new escalation; the closed one stays on record.
- Every action records who did it and when. There is no sign-in yet, so the inbox records every action as "Counselor".
- The seeded escalation's timestamp is days old, so it shows as overdue in the demo.
- **Internal case report.** On any escalation, open or closed, the counselor can generate a report, edit it, save it and download it as a .txt file.
  - It is filled in from the record only, with no model call: student and guardian details, the concern and who flagged it, the family's exact words, a timeline of every step, all messages with the family, and how long the household's messages were paused.
  - The counselor adds their own notes. Regenerating picks up new events and keeps the notes from the last saved version.
  - Every save is a new version; earlier versions are kept and can be viewed read-only.
  - It is a record for the school's files, not the mandated report.
  - Reports are held in memory with the rest of the data, so they are lost on restart or reset.

---

## Not on this branch

- Paging the on-call counselor and the after-hours path
- A check-in button in the UI
- Quiet hours (8am–8pm)
- Voice notes, SMS fallback, language detection (see `CARLINNOTES.md`)
