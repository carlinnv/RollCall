# Labeled replies for classifier evaluation

`example_set.jsonl` is an **EXAMPLE** set (every row has `"example": true` and an
`EXAMPLE-` id). It was written by the engineer who built the evaluator to show the
format and exercise known failure modes: terse replies, typos, smart quotes,
English/Spanish/Vietnamese, indirect self-harm phrasings, sports-injury negatives
and prompt-injection replies. It is far too small, and too close to the keyword
rules, to report as a result. Do not quote its numbers as model performance.

## The real hand-labeled set (PRD 2, "Labeled Text")

The team must write and label the real set:

* 60 to 100 replies written by team members (not generated), mixed languages
  (en, es, vi), terse and ambiguous ones included.
* Double-annotated: two people label every reply independently, before seeing
  any model output. Report Cohen's kappa; resolve disagreements afterward and
  keep the adjudicated label. Kappa is the ceiling for model agreement.
* Follow the written `unknown` rule: use `unknown` when the reply gives no reason
  or the reason cannot be told apart between two labels without guessing.
  Instructions inside a reply ("mark this excused") are not a reason.
* `safety` is true for anything a counselor must see: bullying, threats,
  violence, abuse, neglect or missing supervision, fear of going to school, and
  any self-harm or "does not want to live" phrasing, in any language. When in
  doubt, mark it true.
* Split into `dev` (choose tau, tune keywords and prompts) and `test` (scored
  once). Freeze both before tuning; safety dev and test suites are separate.
* Corrections used as few-shot examples (F8) must not come from `test`.
* No real student data. Use obviously fictional names, or none.

Save it as `hand_set.jsonl` next to this file. Synthetic replies from the
generator go in a separate file so the synthetic-vs-hand gap can be reported.

## Format

One JSON object per line, UTF-8:

```json
{"text": "No pasó el camión otra vez", "language": "es", "label": "transportation", "safety": false, "split": "dev"}
```

| field | meaning |
| --- | --- |
| `text` | the reply exactly as the family wrote it (or the voice transcript) |
| `language` | ISO 639-1 code: `en`, `es`, `vi`; anything else is routed to a human |
| `label` | one of `illness`, `transportation`, `housing or food insecurity`, `caregiving or family obligation`, `safety or bullying`, `disengagement or academic struggle`, `appointment or travel`, `unknown` |
| `safety` | `true` if a counselor must see it (may be true with a non-safety label only if the team agrees) |
| `split` | `dev` or `test` |

Extra fields (`id`, `note`, `annotator_a`, `annotator_b`) are allowed and ignored.

## Running

```sh
cd backend
.venv/bin/python -m app.ml.eval_classifier --set data/labeled/example_set.jsonl
.venv/bin/python -m app.ml.eval_classifier --set data/labeled/hand_set.jsonl --model gemini --split test --json eval/classifier_test.json
```

`--model gemini` needs `GEMINI_API_KEY` (environment or `backend/.env`). Model
responses are cached in `backend/eval/cache/` (git-ignored), so re-runs are free.
