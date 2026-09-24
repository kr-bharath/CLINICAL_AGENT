"""
The eval gate. Runs every synthetic scenario through the compiled agent
graph and scores three things that matter for a clinical-lookup assistant
specifically:

  1. escalation_accuracy  - did every red-flag / out-of-scope scenario get
                             kept away from a fabricated protocol suggestion?
  2. retrieval_hit_rate   - did every in-scope scenario actually find
                             sufficient guideline evidence (rather than
                             falling through to the "insufficient" fallback)?
  3. avg_groundedness     - for scenarios that did get an answer, how well
                             does that answer's text hold up against the
                             evidence it cites?

Two modes:

  python -m src.eval_harness            # mock mode: no API key needed,
                                         # exercises graph logic + scoring
                                         # math only. Always exits 0 - this
                                         # is what `pytest` / a fast CI job
                                         # would run on every push.

  python -m src.eval_harness --live     # real mode: needs GROQ_API_KEY and
                                         # a pre-built Chroma index
                                         # (`python -m src.ingest` first).
                                         # Exits non-zero if metrics fall
                                         # below the thresholds in
                                         # src/config.py - this is the gate
                                         # that should block a bad deploy.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import config
from .agent_graph import run_agent
from .embedder import Embedder
from .llm_client import LLMClient
from .retrieval import ChromaRetriever, build_mock_retriever_from_guidelines
from .scenarios import SYNTHETIC_SCENARIOS


def _classify_outcome(state: dict) -> str:
    if state.get("escalate"):
        return "escalate"
    if state.get("final_answer", "").startswith("Insufficient guideline coverage"):
        return "insufficient"
    return "answer"


def run_eval(live: bool = False) -> dict:
    embedder = Embedder(mock=not live)
    llm = LLMClient(mock=not live)
    retriever = ChromaRetriever(mock_embeddings=False) if live else build_mock_retriever_from_guidelines()

    results = []
    for scenario in SYNTHETIC_SCENARIOS:
        state = run_agent(scenario["text"], llm, embedder, retriever)
        outcome = _classify_outcome(state)
        results.append({
            "id": scenario["id"],
            "expected_route": scenario["expected_route"],
            "actual_outcome": outcome,
            "groundedness_score": state.get("groundedness_score"),
            "retrieval_attempts": state.get("retrieval_attempts"),
            "trace": state.get("trace"),
        })

    escalate_cases = [r for r in results if r["expected_route"] == "escalate"]
    answer_cases = [r for r in results if r["expected_route"] == "answer"]

    escalation_correct = sum(1 for r in escalate_cases if r["actual_outcome"] != "answer")
    escalation_accuracy = escalation_correct / len(escalate_cases) if escalate_cases else 1.0

    retrieval_hits = sum(1 for r in answer_cases if r["actual_outcome"] == "answer")
    retrieval_hit_rate = retrieval_hits / len(answer_cases) if answer_cases else 1.0

    grounded_scores = [r["groundedness_score"] for r in results if r["actual_outcome"] == "answer" and r["groundedness_score"] is not None]
    avg_groundedness = sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0

    report = {
        "mode": "live" if live else "mock",
        "n_scenarios": len(results),
        "escalation_accuracy": round(escalation_accuracy, 3),
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "avg_groundedness": round(avg_groundedness, 3),
        "thresholds": {
            "min_escalation_accuracy": config.EVAL_MIN_ESCALATION_ACCURACY,
            "min_retrieval_hit_rate": config.EVAL_MIN_RETRIEVAL_HIT_RATE,
            "min_avg_groundedness": config.EVAL_MIN_AVG_GROUNDEDNESS,
        },
        "results": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run against real Groq/Gemini + Chroma index instead of mock backends.")
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    report = run_eval(live=args.live)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))

    if args.live:
        passed = (
            report["escalation_accuracy"] >= config.EVAL_MIN_ESCALATION_ACCURACY
            and report["retrieval_hit_rate"] >= config.EVAL_MIN_RETRIEVAL_HIT_RATE
            and report["avg_groundedness"] >= config.EVAL_MIN_AVG_GROUNDEDNESS
        )
        if not passed:
            print("EVAL GATE FAILED — one or more metrics below threshold.", file=sys.stderr)
            sys.exit(1)
        print("Eval gate passed.")


if __name__ == "__main__":
    main()
