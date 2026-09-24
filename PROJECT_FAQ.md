# Clinical Protocol Agent — Project FAQ & Reference

A reference document for explaining this project — to an interviewer, on your
resume, in a README, or to yourself six months from now when you've forgotten
the details. Written as direct Q&A, honestly, without hype.

---

## 1. What is this project, in one sentence?

An agent that looks at a short patient scenario, decides whether it's safe to
answer at all, retrieves the relevant public clinical guideline (WHO/ICMR,
diabetes and hypertension), and only returns a suggestion after checking that
its own answer is actually backed by the text it retrieved — refusing rather
than guessing when it isn't.

**One line for a resume:** *"Built an agentic RAG system for clinical
guideline lookup with a self-verifying groundedness gate and automated CI
eval pipeline, entirely on free-tier infrastructure."*

---

## 2. What problem does it actually solve?

Not "diagnose patients" — a symptom-to-protocol lookup tool answering a
narrower, more honest question: *given this scenario, what does the
published guideline say to do next, and can I prove that from the text I
retrieved?*

The real problem it's a demonstration of solving is **RAG hallucination in a
domain where a wrong answer matters**. Most RAG demos retrieve once, generate
once, and hope the model stuck to the source. This one doesn't hope — it
checks, every time, before returning anything:

- If retrieval doesn't find strong enough evidence, it retries with a
  reformulated query rather than generating from thin material.
- If the generated answer doesn't hold up against the evidence it cites, it
  regenerates with stricter instructions.
- If a scenario looks like a red flag or falls outside its declared scope
  (diabetes/hypertension only), it refuses to generate a suggestion at all
  and routes to "escalate to clinician" instead.

That refusal path is the actual point of the project. A system that always
answers is easy to build and easy to trust wrongly. A system that knows when
*not* to answer, and can show you why, is the harder and more useful thing —
and it's the part interviewers actually want to hear about.

---

## 3. How was it built? (architecture)

```
scenario -> triage --(red flag / out of scope)--> escalate --> END
              |
          (in scope)
              v
          retrieve <---------------------+ (retry, up to 2 attempts)
              |
      sufficient evidence? --no, retries left---+
              |--no, exhausted--> insufficient_evidence --> END
              |--yes
              v
          generate <----------------------+ (regenerate, up to 2 attempts)
              |
        grounded? --no, retries left-------+
              |--no, exhausted--> insufficient_evidence --> END
              |--yes
              v
      safety_guardrail --> END  (attaches citations + disclaimer)
```

Built as a **LangGraph `StateGraph`** — seven nodes, two conditional retry
loops, one hard branch for safety. Not a chain; a graph with actual
decision points.

### Stack (100% free, no subscription anywhere)

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq (`openai/gpt-oss-120b`), Gemini 2.5 Flash as fallback | Free tier, no card. Client tries Groq first, falls over to Gemini on any failure. |
| Embeddings | `sentence-transformers` (local) | Free, no rate limit, no API key — retrieval never depends on a quota. |
| Vector store | Chroma (local, persisted to `.chroma/`) | Zero external service. |
| Orchestration | LangGraph | Conditional edges implement the retry/regenerate loops. |
| Eval gate | Custom harness + GitHub Actions | Free on public repos. |
| Guideline data | Real WHO (2021 hypertension guideline, PEN diabetes protocol) and ICMR (2026 Standard Treatment Workflows) documents, paraphrased into structured excerpts with source citations | Public, current, cited — not fabricated content. |

### Source of truth for guideline text

Four `.md` files under `data/guidelines/`, each with YAML frontmatter
(`doc_id`, `title`, `organization`, `year`, `source_url`) and content chunked
by heading — with a further split into one chunk *per line* where a section
is really several atomic facts (e.g. each blood-pressure grade is its own
chunk, not one blob for "classification"). That granularity decision was a
bug fix, not a design choice made up front — see Q5.

---

## 4. What actually makes this an "agent" instead of a RAG pipeline?

A RAG pipeline retrieves once and generates once. This has three points
where it makes a decision about its own process rather than following a
fixed script:

1. **Retrieval retry loop** — a similarity-score threshold decides if
   evidence is "sufficient." If not, the query gets reformulated (original
   text + topic-specific vocabulary boost) and retried, up to a cap.
2. **Groundedness gate** — every generated answer is split into sentences,
   each compared by embedding similarity against the retrieved evidence.
   Below threshold, it regenerates with a stricter prompt; if that still
   fails, it refuses outright rather than returning a plausible-sounding
   but ungrounded answer.
3. **Triage branch** — red-flag and out-of-scope detection happens
   *before* retrieval, using deterministic keyword rules rather than an
   LLM call. That's a deliberate choice: a safety-critical yes/no gate
   shouldn't depend on a model's classification being consistent call to
   call.

---

## 5. What actually broke during development? (the good interview material)

This is worth memorizing better than the architecture diagram — "tell me
about a bug you found" is one of the most common interview questions, and
every one of these is real, not hypothetical:

1. **Negation-blind keyword matching.** The red-flag detector did plain
   substring matching, so *"no chest pain"* matched `"chest pain"` and
   incorrectly triggered escalation. Fixed with a short negation-cue check
   (`"no "`, `"denies "`, `"without "`, etc.) in a window before the match.
2. **Scope detection missed bare vitals.** A scenario reading `"148/94
   mmHg"` with no literal words "blood pressure" fell through the keyword
   list entirely and got classified out-of-scope. Fixed with a regex
   fallback that recognizes a BP-value pattern directly.
3. **Citation text contaminated the groundedness score.** Evidence passed
   to the generator included inline citation strings, so a "grounded"
   answer's sentences literally contained citation metadata words —
   diluting the very score meant to check faithfulness. Fixed by keeping
   citations entirely separate from the evidence text shown to the model.
4. **Chunking was too coarse.** One markdown section held five different
   BP-grade thresholds as a single chunk, so retrieving for "Grade 1"
   pulled back a chunk equally full of Grade 2 and Grade 3 vocabulary,
   diluting the match. Fixed by splitting line-structured sections into
   one chunk per line.
5. **A LangGraph node name collided with a state key.** A node called
   `"escalate"` and a state field `escalate: bool` conflicted — but only on
   the exact `langgraph` version pip resolved on a clean install, not on
   whatever newer version happened to already be in the dev sandbox. This
   is a real lesson: **test against your pinned `requirements.txt`, not
   whatever's already installed.**
6. **`.env` was never actually loaded.** `python-dotenv` was listed as a
   dependency but no code ever called `load_dotenv()`. The API key sat in
   `.env` doing nothing; every "live" call silently fell through to the
   fallback provider, which then failed for an unrelated, correct reason
   (no Gemini key set) — which brings up bug 7.
7. **Provider errors were being swallowed.** When both LLM providers
   failed, the code only surfaced the *last* error message, hiding what
   actually went wrong with the first (real) provider. Fixed to report
   every provider's failure reason together.
8. **Invalid GitHub Actions syntax.** A job-level `if:` referenced the
   `secrets` context directly (`if: ${{ secrets.GROQ_API_KEY != '' }}`),
   which GitHub explicitly disallows at that scope — it's only readable
   inside a step. Fixed by checking the secret inside a step and passing
   the result forward via job `outputs`.
9. **A dependency-version range was too wide.** `langgraph>=0.2.0,<0.3.0`
   let pip resolve to a version with different enforcement behavior than
   the one tested against (this is bug 5's actual root cause) — pinned to
   an exact version afterward.

None of these were caught by "it looks right" — they were caught by
actually running `pytest`, actually running the CLI, and actually reading
what broke. That process is the demonstrable skill here, more than any
individual line of code.

---

## 6. How do you use it?

```bash
# setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# smoke test, no API key needed
python app.py --mock "Adult with confirmed hypertension, BP 148/94 mmHg on two visits, no chest pain."

# go live
cp .env.example .env   # paste a free Groq key from console.groq.com/keys
python -m src.ingest   # builds the vector index from data/guidelines/*.md
python app.py "58-year-old, blood pressure 210/128 mmHg with severe headache and blurred vision"

# tests and eval
pytest tests/ -v                  # mocked, no key needed, tests control flow
python -m src.eval_harness --live # real model, real index, prints/saves metrics
```

CI (`.github/workflows/eval-gate.yml`) runs the mocked tests on every push
automatically, and the live eval gate once `GROQ_API_KEY` is added as a repo
secret — that's the piece that turns "I tested this once locally" into "this
is continuously verified."

---

## 7. What do the eval numbers actually mean?

Mock-mode baseline across 14 hand-written synthetic scenarios (6 in-scope,
4 red-flag, 4 out-of-scope):

| Metric | Score | What it means |
|---|---|---|
| Escalation accuracy | **1.00** | Every red-flag and out-of-scope scenario was kept away from a fabricated suggestion. This is the safety-critical number. |
| Retrieval hit-rate | **0.83** | 5 of 6 in-scope scenarios found sufficient evidence within the retry budget. The one miss fell through to the safety-net refusal rather than guessing — arguably correct behavior, not just a failure. |
| Avg. groundedness | **0.74** | For answered scenarios, how well the generated text matches its cited evidence (threshold for passing is 0.55). |

Live-mode numbers (real Groq calls) come from your GitHub Actions run —
check the `eval-report` artifact on the `live-eval-gate` job for the current
numbers, since those reflect the real model's behavior, not the mock's.

**Why these numbers aren't a suspicious 100% across the board:** a perfectly
clean eval on every metric usually means the eval was too easy or the test
set was cherry-picked. An 0.83 hit-rate with a documented reason for the miss
is more credible than 1.00 everywhere.

---

## 8. What are its actual limitations? (say this before someone else does)

- **Guideline corpus is paraphrased excerpts, not the full source PDFs.**
  Four documents, ~27 chunks total. It's a demonstration of the pipeline,
  not a comprehensive clinical knowledge base. The README documents exactly
  how to extend it with the full text.
- **Scope is two conditions only** (diabetes, hypertension). Deliberately
  narrow, for depth over breadth.
- **Triage is keyword-based, not a real NLP negation/entity system.** It's
  explicitly documented as a heuristic, not clinical-grade text
  understanding — a determined adversarial phrasing could probably get
  past it. That's a fair thing to say out loud if asked.
- **Groundedness scoring is embedding-similarity, not a full LLM-judge or
  clinical fact-checker.** Cheap and fast on purpose — a stricter (and more
  expensive) LLM-as-judge alternative is a documented, not-yet-built
  extension.
- **All scenarios are synthetic**, by design, for privacy reasons stated
  up front — but that also means it's never been tested against the messy,
  ambiguous phrasing of real clinical notes.
- **No persistence, no auth, no multi-turn conversation, no PHI handling.**
  This is a single-shot lookup demo, not a deployable clinical product, and
  shouldn't be described as one.

---

## 9. Is this actually worth showcasing?

Yes, for what it demonstrates — not for what it diagnoses.

**What it's worth to an interviewer:** it shows you can design a
multi-step agent with real control flow (not just a chain), build an eval
methodology before being asked to, catch your own bugs through actual
testing rather than assuming correctness, debug across sandbox vs. real
environment discrepancies, and fix CI infrastructure when it breaks. That's
a materially stronger portfolio signal than "I called an LLM API and it
worked," and your QA background is exactly what makes the "here's what I
broke and how I found it" section credible rather than performative.

**What it's not worth claiming:** that it's a validated clinical tool, that
its guideline coverage is comprehensive, or that its safety triage would
hold up against adversarial input. Don't oversell it — the honest framing
("portfolio demo of agentic RAG design and eval methodology, scoped to two
conditions, built on synthetic data only") is more credible than a bigger
claim, and it's also just true.

---

## 10. Likely interview questions and how to answer them

- **"Walk me through the architecture."** → Use the diagram in Q3. Lead
  with the two loops and the triage branch — that's the agentic part, not
  the retrieval-then-generate part.
- **"What's the hardest bug you hit?"** → Pick #5 or #9 from Q5 (the
  LangGraph state-key collision, or the GitHub Actions `secrets` context
  restriction) — both are genuinely non-obvious framework/platform
  constraints, not typos, and both required reading actual error messages
  and framework source/docs rather than guessing.
- **"How do you know it's not hallucinating?"** → The groundedness gate,
  explained concretely: sentence-level embedding comparison against
  retrieved evidence, with a hard threshold and a regenerate-then-refuse
  fallback. Mention the numbers from Q7.
- **"Why not just use a bigger/commercial guideline database?"** → Scope
  and cost control for a demo; the pipeline is agnostic to corpus size —
  swapping in the full WHO/ICMR PDFs is a documented extension, not a
  redesign.
- **"Why deterministic rules for triage instead of asking the LLM?"** →
  A safety-critical yes/no gate shouldn't depend on a model being
  consistent call to call; deterministic rules are auditable and testable
  in a way an LLM classification isn't, at the cost of being less flexible
  to phrasing (acknowledge the negation bug here — it shows you understand
  the trade-off, not just picked a side).
- **"What would you change with more time?"** → See Q11.
- **"Is this HIPAA-compliant / production-ready?"** → No, and say so
  plainly — no PHI handling, no auth, no persistence, synthetic data only,
  by design. It's a methodology demo.

---

## 11. What would you actually do next, with more time?

Roughly in order of value-to-effort:

1. **Full-text ingestion** of the real WHO/ICMR PDFs instead of paraphrased
   excerpts — the ingestion pipeline already supports this extension path.
2. **An LLM-as-judge groundedness mode** as a stricter, opt-in alternative
   to the embedding-similarity heuristic, with the two compared against
   each other on the same eval set.
3. **A negation-aware or small-model-based triage classifier** to replace
   the keyword heuristic, with the current keyword version kept as a fast
   first-pass filter.
4. **Multi-turn scenario handling** — right now every call is stateless;
   real clinical intake is a conversation, not a single message.
5. **A second corpus** (e.g. a third condition) to test whether the
   architecture generalizes without changes, which is the real test of
   whether this is a reusable pattern or a one-off.

---

## 12. Quick-reference cheat sheet

```bash
# Run one scenario, see the full decision trace
python app.py "<scenario text>"                 # live
python app.py --mock "<scenario text>"           # no API key needed

# Build/rebuild the vector index after editing data/guidelines/*.md
python -m src.ingest

# Full test suite (fast, mocked, CI-safe)
pytest tests/ -v

# Eval harness
python -m src.eval_harness             # mock, always exits 0
python -m src.eval_harness --live      # real model, exits non-zero on gate failure

# Free API keys
Groq:   https://console.groq.com/keys       (primary)
Gemini: https://aistudio.google.com/apikey  (fallback)
```
