"""
Run one scenario through the agent and print its full decision trace.

    python app.py "52-year-old, BP 148/94 mmHg on two visits, no red flags"
    python app.py --mock "52-year-old, BP 148/94 mmHg on two visits"   # no API key / no ingested index needed

Real mode requires GROQ_API_KEY (and, optionally, GEMINI_API_KEY as a
fallback) in the environment, and a Chroma index already built with
`python -m src.ingest`.
"""
import argparse
import json
import sys

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.agent_graph import run_agent
from src.retrieval import ChromaRetriever, build_mock_retriever_from_guidelines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Free-text synthetic patient scenario.")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM + embeddings (no API key, no ingested index needed).")
    args = parser.parse_args()

    embedder = Embedder(mock=args.mock)
    llm = LLMClient(mock=args.mock)
    retriever = build_mock_retriever_from_guidelines() if args.mock else ChromaRetriever()

    state = run_agent(args.scenario, llm, embedder, retriever)

    print("\n--- TRACE ---")
    for line in state.get("trace", []):
        print(f"  {line}")

    print("\n--- FINAL ANSWER ---")
    print(state.get("final_answer", "(no answer produced)"))

    if state.get("citations"):
        print("\n--- CITATIONS ---")
        for c in state["citations"]:
            print(f"  - {c}")


if __name__ == "__main__":
    sys.exit(main())
