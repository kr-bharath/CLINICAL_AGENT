"""
Central configuration. Every threshold the agent's control flow depends on
lives here, so the eval harness and the graph never disagree about what
"good enough" means.
"""
import os

from dotenv import load_dotenv

# Every other module gets to GROQ_API_KEY / GEMINI_API_KEY via plain
# os.getenv() calls, not through a constant defined here — so this has to
# run, once, before any of those calls happen. Since every module in this
# project imports `config` (directly or transitively) before touching an
# API key, putting the load here at import time is what actually makes
# `.env` take effect anywhere in the codebase.
load_dotenv()

# --- LLM ---
# Groq deprecated llama-3.1-8b-instant / llama-3.3-70b-versatile on 2026-08-16.
# These are the current recommended free-tier replacements.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.2

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector store ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
COLLECTION_NAME = "clinical_guidelines"
GUIDELINES_DIR = os.getenv("GUIDELINES_DIR", "data/guidelines")

# --- Retrieval / agent control flow ---
RETRIEVAL_TOP_K = 4
SUFFICIENCY_SCORE_THRESHOLD = 0.35   # min top similarity score to call evidence "sufficient"
MAX_RETRIEVAL_ATTEMPTS = 2           # re-query loop cap

# --- Generation / groundedness gate ---
GROUNDEDNESS_THRESHOLD = 0.55        # min avg sentence-to-evidence similarity
MAX_GENERATION_ATTEMPTS = 2          # regenerate loop cap

# --- Scope ---
IN_SCOPE_KEYWORDS = {
    "diabetes": ["diabetes", "diabetic", "glucose", "hba1c", "polyuria", "polydipsia", "hyperglyc", "hypoglyc", "insulin", "metformin", "ketoacidosis", "dka"],
    "hypertension": ["hypertension", "blood pressure", "bp ", "systolic", "diastolic", "htn", "hypertensive"],
}

# Appended (not substituted) to the original query on a retrieval retry.
# Keeps the scenario's own specific vocabulary (vitals, values) while adding
# the topic terms guideline text is written in, so a re-query doesn't throw
# away the one thing that made the first query specific.
TOPIC_QUERY_BOOST = {
    "diabetes": "diabetes glucose hba1c metformin glycemic management screening treatment",
    "hypertension": "hypertension blood pressure systolic diastolic pharmacological treatment threshold target grade classification",
}

RED_FLAG_PATTERNS = [
    "chest pain", "severe headache", "breathless", "altered sensorium", "confusion",
    "fruity breath", "rapid breathing", "visual disturbance", "blurred vision and headache",
    "loss of consciousness", "seizure", "persistent vomiting",
]

# Lightweight negation guard for the red-flag matcher below: plain substring
# matching on "chest pain" also fires on "no chest pain" or "denies chest
# pain", which would over-escalate. This is a heuristic (checked within a
# short character window immediately before the match), not real clinical
# NLP negation detection (e.g. negspaCy) — good enough for a demo triage
# rule, documented as a known limitation rather than hidden.
NEGATION_CUES = ["no ", "not ", "denies ", "denying ", "without ", "negative for ", "absence of ", "ruled out "]
NEGATION_WINDOW_CHARS = 15

# --- Eval gate thresholds (used by src/eval_harness.py in CI) ---
EVAL_MIN_ESCALATION_ACCURACY = 0.90
EVAL_MIN_AVG_GROUNDEDNESS = 0.55
EVAL_MIN_RETRIEVAL_HIT_RATE = 0.80
