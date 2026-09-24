# Clinical protocol agent (portfolio project)

[![eval-gate](https://github.com/kr-bharath/CLINICAL_AGENT/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/kr-bharath/CLINICAL_AGENT/actions/workflows/eval-gate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A symptom-to-protocol assistant for **diabetes and hypertension**, built as an
*agent* rather than a single-shot RAG pipeline: it can retry a weak
retrieval, refuse to answer when its own draft isn't grounded in the
retrieved guideline text, and route red-flag or out-of-scope cases to a
clinician instead of guessing.

**This is a portfolio / research demo. It runs only on synthetic scenarios,
never real patient data, and its output is not a diagnosis or a substitute
for a qualified clinician.**

📄 **[Full project FAQ](PROJECT_FAQ.md)** — architecture walkthrough, every
real bug hit during development and how it was fixed, eval numbers
explained, honest limitations, and an interview-prep Q&A section.

## Why an agent, not a RAG pipeline

A plain RAG pipeline retrieves once and generates once. Two things here
make it an agent instead:

- **Retrieval loop** — if the retrieved evidence isn't similar enough to the
  query, the agent reformulates the query and retries (up to
  `MAX_RETRIEVAL_ATTEMPTS`) before ever reaching generation.
- **Groundedness gate** — after a draft answer is written, every sentence is
  checked against the retrieved evidence. Below threshold, the agent either
  regenerates with a stricter prompt or gives up and says so, rather than
  returning an ungrounded answer.
- **Triage branch** — red-flag presentations and anything outside
  diabetes/hypertension are routed to escalation *before* retrieval ever
  runs, using deterministic keyword rules rather than trusting an LLM
  classification for a safety-critical decision.

See `src/agent_graph.py` for the LangGraph implementation, or run
`python app.py --mock "<scenario>"` and read the printed trace.

## Architecture

```
scenario -> triage --(red flag / out of scope)--> escalate --> END
              |
          (in scope)
              v
          retrieve <---------------------+ (retry, up to MAX_RETRIEVAL_ATTEMPTS)
              |
      sufficient? --no, retries left------+
              |--no, exhausted--> insufficient_evidence --> END
              |--yes
              v
          generate <----------------------+ (regenerate, up to MAX_GENERATION_ATTEMPTS)
              |
        grounded? --no, retries left-------+
              |--no, exhausted--> insufficient_evidence --> END
              |--yes
              v
      safety_guardrail --> END  (adds citations + disclaimer)
```

## Free stack

| Layer | What's used | Notes |
|---|---|---|
| LLM | Groq, `openai/gpt-oss-120b` (free tier) | Falls back to Google AI Studio's Gemini 2.5 Flash free tier if Groq errors/rate-limits. Groq retired `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` on 2026-08-16 — `src/config.py` already points at the current recommended replacement. |
| Embeddings | `sentence-transformers` (local) | Free, no rate limit, no API key. |
| Vector store | Chroma (local, persisted to `.chroma/`) | Zero external service. |
| Orchestration | LangGraph | Conditional edges implement the retry/regenerate loops. |
| Eval gate | Custom harness + GitHub Actions | Free on public repos. |

No paid subscription is required anywhere in this stack.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your free GROQ_API_KEY
```

Get a free Groq key (no card required): https://console.groq.com/keys

### Try it without any API key at all

```bash
python app.py --mock "52-year-old male, repeat clinic readings 148/94 mmHg on two visits, no chest pain or visual symptoms."
```

This uses the deterministic mock LLM and a local hashed embedding, so it
needs no network access — good for a first smoke test or for running in a
network-restricted environment.

### Run for real

```bash
python -m src.ingest              # chunk + embed data/guidelines/*.md into Chroma
python app.py "58-year-old, blood pressure 210/128 mmHg with severe headache and blurred vision"
```

## Testing and the eval gate

```bash
pytest tests/ -v                       # fast, mocked, no API key — tests the control flow
python -m src.eval_harness             # same, but reports escalation/groundedness/hit-rate numbers
python -m src.eval_harness --live      # real model + real index; exits non-zero if metrics
                                        # in src/config.py aren't met — wire GROQ_API_KEY as a
                                        # repo secret and this becomes a CI gate (see
                                        # .github/workflows/eval-gate.yml)
```

The eval harness scores three things, on 14 hand-written synthetic
scenarios (`src/scenarios.py`) spanning in-scope diabetes/hypertension
cases, red-flag emergencies, and out-of-scope presentations:

- **escalation_accuracy** — did every red-flag/out-of-scope case avoid
  getting a fabricated protocol suggestion?
- **retrieval_hit_rate** — did every in-scope case find sufficient evidence?
- **avg_groundedness** — for answered cases, how well does the answer text
  actually match its cited evidence?

## Extending with the full guideline text

`data/guidelines/*.md` currently holds paraphrased, structured excerpts of
four real public documents (see each file's frontmatter for the exact
source and URL):

- WHO, *Guideline for the pharmacological treatment of hypertension in
  adults* (2021)
- WHO, *Package of Essential Noncommunicable Disease Interventions* (PEN) —
  diabetes component
- ICMR/DHR, *Standard Treatment Workflow — Hypertension in Adults* (2026)
- ICMR/DHR, *Standard Treatment Workflow — Diabetes Mellitus Type 2*

To ground the agent in the *full* text rather than these excerpts:
download the PDFs from the `source_url` in each file's frontmatter, extract
text (the `pdf-reading` approach of chunk-by-section applies directly), and
either replace these `.md` files or extend `src/ingest.py` to also read
PDFs. The chunking, embedding, retrieval, and eval-gate code all stay the
same — they operate on whatever's in `data/guidelines/`.

## Project layout

```
LICENSE                  MIT
PROJECT_FAQ.md           architecture, real bugs found, eval numbers, interview Q&A
data/guidelines/*.md     structured guideline excerpts (source + citation in frontmatter)
src/config.py            every threshold the control flow depends on
src/llm_client.py        Groq -> Gemini fallback, or a mock backend for tests
src/embedder.py          local sentence-transformers, or a mock backend for tests
src/ingest.py            chunk + embed guidelines into Chroma
src/retrieval.py         Chroma-backed retriever + a dependency-free in-memory one for tests
src/groundedness.py      the eval-gate scoring function
src/agent_graph.py       the LangGraph agent itself
src/scenarios.py         synthetic patient scenarios (never real data)
src/eval_harness.py      runs scenarios through the graph, scores, gates CI
app.py                   CLI: run one scenario, print the trace
tests/test_agent_graph.py  control-flow tests, no API key needed
.github/workflows/eval-gate.yml  fast mocked tests + optional live gate
```
