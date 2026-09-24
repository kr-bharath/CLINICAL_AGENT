"""
The agent, expressed as a LangGraph StateGraph.

    synthetic scenario
            |
      scope & triage  ----(red flag / out of scope)----> escalate --> END
            |
     (in scope)
            |
      retrieve guidelines <---------------------+
            |                                    | (loop: re-query, up to
      sufficiency check                          |  MAX_RETRIEVAL_ATTEMPTS)
            |-- insufficient, retries left -------+
            |-- insufficient, retries exhausted --> insufficient_evidence --> END
            |-- sufficient
            v
      draft suggested step <----------------------+
            |                                       | (loop: regenerate, up to
      groundedness gate                             |  MAX_GENERATION_ATTEMPTS)
            |-- ungrounded, retries left -----------+
            |-- ungrounded, retries exhausted -----> insufficient_evidence --> END
            |-- grounded
            v
      safety guardrail --> output --> END

Every node appends a one-line note to `state["trace"]`, which is what makes
this an *agent* trace rather than a black box: you can print `trace` and
see exactly which path a given scenario took and why.
"""
from __future__ import annotations
import re
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from . import config
from .embedder import Embedder
from .groundedness import score_groundedness
from .llm_client import LLMClient
from .retrieval import RetrievalResult, is_sufficient

_BP_VALUE_RE = re.compile(r"\d{2,3}\s*/\s*\d{2,3}\s*mm\s*hg")


class AgentState(TypedDict, total=False):
    scenario_text: str
    in_scope: bool
    escalate: bool
    escalate_reason: str
    retrieval_attempts: int
    retrieved: List[RetrievalResult]
    generation_attempts: int
    draft_answer: str
    groundedness_score: float
    grounded: bool
    final_answer: str
    citations: List[str]
    trace: List[str]


def _detect_red_flag(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern in config.RED_FLAG_PATTERNS:
        idx = lowered.find(pattern)
        if idx == -1:
            continue
        window_start = max(0, idx - config.NEGATION_WINDOW_CHARS)
        preceding = lowered[window_start:idx]
        if any(cue in preceding for cue in config.NEGATION_CUES):
            continue  # negated mention (e.g. "no chest pain") - not a red flag
        return pattern
    return None


def _detect_scope(text: str) -> Optional[str]:
    lowered = text.lower()
    for topic, keywords in config.IN_SCOPE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return topic
    # Keyword lists miss a bare vitals mention like "148/94 mmHg" with no
    # accompanying word "blood pressure"/"BP" — catch that pattern directly
    # rather than relying on wording alone.
    if _BP_VALUE_RE.search(lowered):
        return "hypertension"
    return None


def build_graph(llm: LLMClient, embedder: Embedder, retriever):
    """
    `retriever` is anything with a `.retrieve(query, top_k) -> List[RetrievalResult]`
    method — a ChromaRetriever or an InMemoryRetriever both satisfy this.
    """

    def triage_node(state: AgentState) -> AgentState:
        text = state["scenario_text"]
        trace = state.get("trace", [])
        red_flag = _detect_red_flag(text)
        topic = _detect_scope(text)

        if red_flag:
            trace.append(f"triage: red-flag pattern matched ('{red_flag}') -> escalate")
            return {**state, "escalate": True, "in_scope": bool(topic),
                    "escalate_reason": f"Presentation matches a red-flag pattern ('{red_flag}') requiring urgent clinical attention rather than a guideline-based suggestion.",
                    "trace": trace}
        if not topic:
            trace.append("triage: no diabetes/hypertension keywords matched -> out of scope, escalate")
            return {**state, "escalate": True, "in_scope": False,
                    "escalate_reason": "Scenario does not match this assistant's scope (diabetes or hypertension protocols).",
                    "trace": trace}

        trace.append(f"triage: in-scope ({topic}), no red flags -> proceed to retrieval")
        return {**state, "escalate": False, "in_scope": True, "trace": trace}

    def escalate_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        trace.append("escalate: routing to clinician, no protocol suggestion generated")
        return {
            **state,
            "final_answer": f"This case should be escalated to a clinician rather than handled by protocol lookup. Reason: {state.get('escalate_reason', 'unspecified')}",
            "citations": [],
            "trace": trace,
        }

    def retrieve_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        attempts = state.get("retrieval_attempts", 0) + 1
        query = state["scenario_text"]
        if attempts > 1:
            # Re-query reformulation: keep the scenario's own vocabulary
            # (vitals, values) and append topic terms the guideline text is
            # actually written in, rather than throwing the specific query
            # away for a generic one.
            topic = _detect_scope(state["scenario_text"]) or ""
            boost = config.TOPIC_QUERY_BOOST.get(topic, "management guideline recommendation")
            query = f"{query} {boost}"
        results = retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K)
        top_score = results[0].score if results else 0.0
        trace.append(f"retrieve (attempt {attempts}): query='{query[:60]}...' top_score={top_score:.2f} n_results={len(results)}")
        return {**state, "retrieved": results, "retrieval_attempts": attempts, "trace": trace}

    def sufficiency_router(state: AgentState) -> str:
        if is_sufficient(state.get("retrieved", [])):
            return "generate"
        if state.get("retrieval_attempts", 0) < config.MAX_RETRIEVAL_ATTEMPTS:
            return "retrieve"
        return "insufficient_evidence"

    def generate_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        attempts = state.get("generation_attempts", 0) + 1
        evidence = state.get("retrieved", [])
        # Deliberately plain guideline text only, no citation metadata mixed
        # in — citations are attached independently by safety_guardrail_node
        # from `state["retrieved"]`. Keeping this block pure means the
        # groundedness gate compares the answer against the same raw text
        # the model actually saw, with nothing extra to dilute the match.
        evidence_block = "\n".join(f"- {r.text}" for r in evidence)

        strictness = (
            "Use ONLY the evidence below. Every sentence you write must be directly "
            "supported by at least one evidence line. Do not add clinical detail that "
            "is not present in the evidence."
        )
        if attempts > 1:
            strictness += " Your previous attempt included unsupported claims — be more conservative this time."

        system_prompt = (
            "You are a clinical protocol lookup assistant. You suggest a next step based only on the "
            "evidence given to you. You never diagnose, and you never state a source name yourself — "
            "citations are attached separately by the system, so just write the clinical content."
        )
        user_prompt = (
            f"{strictness}\n\nPatient scenario (synthetic): {state['scenario_text']}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Write a short suggested next step grounded in the evidence above."
        )
        response = llm.call(system_prompt, user_prompt)
        trace.append(f"generate (attempt {attempts}, provider={response.provider}): drafted answer")
        return {**state, "draft_answer": response.text, "generation_attempts": attempts, "trace": trace}
    def groundedness_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        report = score_groundedness(state["draft_answer"], state.get("retrieved", []), embedder)
        grounded = report.score >= config.GROUNDEDNESS_THRESHOLD
        trace.append(f"groundedness gate: score={report.score:.2f} threshold={config.GROUNDEDNESS_THRESHOLD} grounded={grounded}")
        return {**state, "groundedness_score": report.score, "grounded": grounded, "trace": trace}

    def groundedness_router(state: AgentState) -> str:
        if state.get("grounded"):
            return "safety_guardrail"
        if state.get("generation_attempts", 0) < config.MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "insufficient_evidence"

    def safety_guardrail_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        citations = [r.citation() for r in state.get("retrieved", [])]
        disclaimer = (
            "\n\n[This is a guideline-lookup suggestion generated from a synthetic scenario for a "
            "portfolio demo. It is not a diagnosis and is not a substitute for a qualified clinician.]"
        )
        trace.append("safety guardrail: attached citations + mandatory disclaimer")
        return {
            **state,
            "final_answer": state["draft_answer"] + disclaimer,
            "citations": citations,
            "trace": trace,
        }

    def insufficient_evidence_node(state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        trace.append("insufficient_evidence: retries exhausted without clearing sufficiency/groundedness threshold -> refusing rather than guessing")
        return {
            **state,
            "final_answer": "Insufficient guideline coverage to ground a confident suggestion for this scenario. Recommend human clinical review.",
            "citations": [r.citation() for r in state.get("retrieved", [])],
            "trace": trace,
        }

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("escalate_case", escalate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("groundedness_gate", groundedness_node)
    graph.add_node("safety_guardrail", safety_guardrail_node)
    graph.add_node("insufficient_evidence", insufficient_evidence_node)

    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", lambda s: "escalate" if s.get("escalate") else "retrieve",
                                 {"escalate": "escalate_case", "retrieve": "retrieve"})
    graph.add_edge("escalate_case", END)
    graph.add_conditional_edges("retrieve", sufficiency_router,
                                 {"generate": "generate", "retrieve": "retrieve", "insufficient_evidence": "insufficient_evidence"})
    graph.add_edge("generate", "groundedness_gate")
    graph.add_conditional_edges("groundedness_gate", groundedness_router,
                                 {"safety_guardrail": "safety_guardrail", "generate": "generate", "insufficient_evidence": "insufficient_evidence"})
    graph.add_edge("safety_guardrail", END)
    graph.add_edge("insufficient_evidence", END)

    return graph.compile()


def run_agent(scenario_text: str, llm: LLMClient, embedder: Embedder, retriever) -> AgentState:
    app = build_graph(llm, embedder, retriever)
    initial_state: AgentState = {"scenario_text": scenario_text, "trace": []}
    return app.invoke(initial_state)
