"""
These tests exercise the agent's *decisions*, not any live model's text
quality — that's what the mock LLM/embedder backends are for. They run in
CI on every push, need no API key, and no prior `python -m src.ingest`.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent_graph import run_agent
from src.embedder import Embedder
from src.llm_client import LLMClient
from src.retrieval import build_mock_retriever_from_guidelines


def _agent_components():
    return LLMClient(mock=True), Embedder(mock=True), build_mock_retriever_from_guidelines()


def test_red_flag_routes_to_escalate_without_generating_advice():
    llm, embedder, retriever = _agent_components()
    state = run_agent(
        "58-year-old, blood pressure 210/128 mmHg with severe headache and blurred vision, appears distressed.",
        llm, embedder, retriever,
    )
    assert state["escalate"] is True
    assert "draft_answer" not in state
    assert "escalate" in " ".join(state["trace"])


def test_out_of_scope_routes_to_escalate():
    llm, embedder, retriever = _agent_components()
    state = run_agent(
        "24-year-old with a twisted ankle after a fall while playing football, mild swelling.",
        llm, embedder, retriever,
    )
    assert state["escalate"] is True
    assert state["in_scope"] is False


def test_in_scope_case_reaches_safety_guardrail_and_is_grounded():
    llm, embedder, retriever = _agent_components()
    state = run_agent(
        "Adult with confirmed hypertension: repeat clinic blood pressure readings 148/94 mmHg (grade 1) "
        "on two visits, currently untreated, no chest pain, no visual disturbance.",
        llm, embedder, retriever,
    )
    assert state["escalate"] is False
    assert state["grounded"] is True
    assert state["groundedness_score"] >= 0.55
    assert len(state["citations"]) > 0
    assert "disclaimer" not in state["final_answer"].lower() or "not a diagnosis" in state["final_answer"].lower()


def test_retrieval_loop_retries_before_giving_up():
    llm, embedder, retriever = _agent_components()
    state = run_agent(
        "Type 2 diabetes on metformin and lifestyle modification for six months, HbA1c 8.2 percent, "
        "glycemic target not met, no hypoglycemia, requesting next step in management escalation.",
        llm, embedder, retriever,
    )
    # Whatever the outcome, the loop must have made at least one attempt and
    # must never exceed the configured cap.
    from src import config
    assert 1 <= state["retrieval_attempts"] <= config.MAX_RETRIEVAL_ATTEMPTS


def test_insufficient_evidence_fallback_when_retriever_is_empty():
    from src.retrieval import InMemoryRetriever

    llm = LLMClient(mock=True)
    embedder = Embedder(mock=True)
    empty_retriever = InMemoryRetriever(chunks=[], embedder=embedder)

    state = run_agent(
        "48-year-old with fasting plasma glucose 152 mg/dL on two occasions, mild polyuria.",
        llm, embedder, empty_retriever,
    )
    assert state["final_answer"].startswith("Insufficient guideline coverage")
