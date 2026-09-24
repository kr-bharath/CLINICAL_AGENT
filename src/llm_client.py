"""
LLM client with a real free-tier backend (Groq, falling back to Google AI
Studio's Gemini free tier if Groq errors or rate-limits) and a deterministic
mock backend so the rest of the system — and CI — can be tested without any
API key at all.

This is intentionally the *only* place that talks to a network LLM. Every
other module depends on this interface, not on a specific provider.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from . import config


@dataclass
class LLMResponse:
    text: str
    provider: str


class LLMClient:
    def __init__(self, mock: bool = False):
        self.mock = mock or os.getenv("CLINICAL_AGENT_MOCK", "").lower() in {"1", "true", "yes"}
        self._groq_client = None
        self._gemini_configured = False

    # -- lazy provider setup, so importing this module never requires the
    #    optional SDKs or API keys to be present --
    def _get_groq(self):
        if self._groq_client is None:
            from groq import Groq  # local import: optional dependency
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_groq()
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai  # local import: optional dependency
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        if not self._gemini_configured:
            genai.configure(api_key=api_key)
            self._gemini_configured = True
        model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=system_prompt)
        result = model.generate_content(user_prompt)
        return result.text

    def _call_mock(self, system_prompt: str, user_prompt: str) -> str:
        """
        Deterministic stand-in used by tests and by `--mock` CLI runs.
        It does the one thing that matters for testing the *agent's* logic:
        it lifts sentences directly out of whatever evidence was placed in
        the prompt, so a grounded answer is trivially grounded and the
        groundedness gate's *pass* path is exercised. To exercise the
        *fail* path deterministically, tests pass evidence-free prompts.

        The picked evidence sentences are kept in their own sentence
        boundaries, separate from the "Suggested next step." lead-in —
        fusing boilerplate text into the same sentence as real evidence
        would dilute that sentence's word-overlap score against the
        evidence it's supposed to be trivially grounded in.
        """
        marker = "EVIDENCE:\n"
        if marker in user_prompt:
            evidence_block = user_prompt.split(marker, 1)[1]
            sentences = [s.strip() for s in evidence_block.replace("\n", " ").split(".") if s.strip()]
            picked = sentences[:2] if sentences else []
            if picked:
                body = ". ".join(s.rstrip(".") for s in picked) + "."
                return f"{body} [MOCK RESPONSE — grounded in retrieved evidence]"
        return "Unable to ground a specific recommendation in the retrieved evidence. [MOCK RESPONSE]"

    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if self.mock:
            return LLMResponse(text=self._call_mock(system_prompt, user_prompt), provider="mock")

        errors = []
        for provider_name, fn in (("groq", self._call_groq), ("gemini", self._call_gemini)):
            try:
                return LLMResponse(text=fn(system_prompt, user_prompt), provider=provider_name)
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider failure should fail over
                errors.append(f"{provider_name}: {exc}")
                continue
        raise RuntimeError("All LLM providers failed:\n  " + "\n  ".join(errors))
