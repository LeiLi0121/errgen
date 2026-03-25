"""
Base analysis agent.

All section-level analysis agents inherit from BaseAnalysisAgent, which
provides the shared evidence-first generation contract:

  1. Receive evidence chunks and calculation results.
  2. Call LLM with JSON-mode prompt that demands chunk_id citations.
  3. Validate that cited chunk IDs actually exist in the evidence pool.
  4. Return a list of AnalysisParagraph objects with citations attached.

The base class does NOT do any retry / verification – that is handled by
the pipeline's checker/reviser loop.
"""

from __future__ import annotations

import logging
from typing import Any

from errgen.llm import build_messages, chat_json
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    Citation,
    EvidenceChunk,
    SourceType,
    VerificationStatus,
)
from errgen.prompt_aliases import (
    PromptAliasMaps,
    build_prompt_alias_maps,
    extract_aliases_from_text,
    strip_inline_aliases,
    unique_preserve_order,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared system-prompt fragment that ALL analysis agents include
# ---------------------------------------------------------------------------

_EVIDENCE_RULES = """
CRITICAL EVIDENCE RULES:
1. Every factual claim MUST be supported by at least one cited chunk_id.
2. Every number or percentage MUST come from a cited chunk OR a cited calc_id.
3. Do NOT invent, estimate, or extrapolate beyond the provided evidence.
4. If the evidence is insufficient, explicitly state that in the paragraph.
5. Use precise figures from the evidence (e.g. "$60.9B" not "about $60B").
6. Do NOT reference events after the as-of date.
7. Avoid overclaiming: use "suggests", "indicates", "according to" rather than "will" or "definitely".
"""

_OUTPUT_FORMAT = """
Return a JSON object with this exact structure:
{
  "paragraphs": [
    {
      "text": "The paragraph text. Be specific, analytical, and fully grounded.",
      "chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "calc_ids": ["calc_id_1"]
    }
  ]
}

- chunk_ids: list of prompt chunk references such as C001, C002.
- calc_ids: list of prompt calc references such as K001, K002.
- Each paragraph must have at least one chunk_id.
- Write 2–4 focused paragraphs per section unless stated otherwise.
- Do NOT embed chunk or calc references inline in the paragraph text.
"""


def _format_chunks_for_prompt(
    chunks: list[EvidenceChunk],
    alias_maps: PromptAliasMaps,
    max_chars: int = 12000,
) -> str:
    """Serialize evidence chunks to a numbered reference block for the prompt."""
    lines: list[str] = []
    total = 0
    for chunk in chunks:
        alias = alias_maps.chunk_id_to_alias.get(chunk.chunk_id, chunk.chunk_id)
        entry = (
            f"[CHUNK_REF: {alias}]\n"
            f"Type: {chunk.source_type.value} | Period: {chunk.period or 'N/A'} | "
            f"Field: {chunk.field_name or 'N/A'}\n"
            f"{chunk.text}\n---\n"
        )
        if total + len(entry) > max_chars:
            lines.append("[...additional chunks truncated for length...]")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def _format_calcs_for_prompt(
    calcs: list[CalculationResult],
    alias_maps: PromptAliasMaps,
) -> str:
    """Serialize calculation results to a reference block for the prompt."""
    if not calcs:
        return "(no calculations available)"
    lines: list[str] = []
    for calc in calcs:
        alias = alias_maps.calc_id_to_alias.get(calc.calc_id, calc.calc_id)
        if calc.error:
            lines.append(
                f"[CALC_REF: {alias}]\n"
                f"Operation: {calc.operation} | ERROR: {calc.error}\n---"
            )
        else:
            result_str = (
                f"{calc.result:.4f}" if isinstance(calc.result, float) else str(calc.result)
            )
            lines.append(
                f"[CALC_REF: {alias}]\n"
                f"Description: {calc.description}\n"
                f"Formula: {calc.formula_description}\n"
                f"Result: {result_str}\n---"
            )
    return "\n".join(lines)


class BaseAnalysisAgent:
    """
    Base class for all section analysis agents.

    Subclasses must implement `section_name` (str) and
    `_build_user_prompt(...)` → str.
    """

    section_name: str = "generic"

    def generate(
        self,
        ticker: str,
        request_context: str,
        chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        as_of_date: str | None = None,
        model: str | None = None,
    ) -> list[AnalysisParagraph]:
        """
        Generate analysis paragraphs for this section.

        Returns AnalysisParagraph objects with VerificationStatus.FAIL
        (to be updated by the checker/reviser loop).
        """
        if not chunks:
            logger.warning(
                "%s: no evidence chunks available for %s – skipping generation.",
                self.__class__.__name__, ticker,
            )
            return []

        alias_maps = build_prompt_alias_maps(chunks, calc_results)
        chunk_block = _format_chunks_for_prompt(chunks, alias_maps)
        calc_block = _format_calcs_for_prompt(calc_results, alias_maps)

        system_prompt = self._build_system_prompt(as_of_date)
        user_prompt = self._build_user_prompt(
            ticker=ticker,
            request_context=request_context,
            chunk_block=chunk_block,
            calc_block=calc_block,
            as_of_date=as_of_date,
        )

        messages = build_messages(system_prompt, user_prompt)

        try:
            raw: dict[str, Any] = chat_json(messages, model=model)
        except Exception as exc:
            logger.error(
                "%s: LLM generation failed for %s: %s",
                self.__class__.__name__, ticker, exc,
            )
            return []

        paragraphs = self._parse_response(
            raw=raw,
            section_name=self.section_name,
            alias_maps=alias_maps,
            chunk_map={c.chunk_id: c for c in chunks},
        )

        logger.info(
            "%s: generated %d paragraphs for %s",
            self.__class__.__name__, len(paragraphs), ticker,
        )
        return paragraphs

    def _build_system_prompt(self, as_of_date: str | None) -> str:
        date_clause = (
            f"\nAs-of date: {as_of_date}. Do NOT reference any events after this date."
            if as_of_date
            else ""
        )
        return (
            f"You are a senior equity research analyst writing the "
            f"'{self.section_name}' section of a professional research report.{date_clause}\n"
            f"{_EVIDENCE_RULES}\n"
            f"{_OUTPUT_FORMAT}"
        )

    def _build_user_prompt(
        self,
        ticker: str,
        request_context: str,
        chunk_block: str,
        calc_block: str,
        as_of_date: str | None,
    ) -> str:
        """Override in subclass for section-specific framing."""
        return (
            f"Ticker: {ticker}\n"
            f"Report context: {request_context}\n\n"
            f"=== EVIDENCE CHUNKS ===\n{chunk_block}\n\n"
            f"=== CALCULATION RESULTS ===\n{calc_block}\n\n"
            f"Write the '{self.section_name}' section."
        )

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        section_name: str,
        alias_maps: PromptAliasMaps,
        chunk_map: dict[str, EvidenceChunk],
    ) -> list[AnalysisParagraph]:
        """
        Parse LLM JSON response into AnalysisParagraph objects.

        Validates that cited IDs exist. Strips non-existent IDs with a warning
        but does NOT drop the whole paragraph – the checker will flag missing
        citations as issues in the verification pass.
        """
        paragraphs: list[AnalysisParagraph] = []
        raw_paragraphs: list[dict] = raw.get("paragraphs", [])

        for item in raw_paragraphs:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            text_chunk_aliases, text_calc_aliases = extract_aliases_from_text(text)
            cleaned_text = strip_inline_aliases(text) or text

            raw_chunk_aliases = unique_preserve_order(
                list(item.get("chunk_ids", [])) + text_chunk_aliases
            )
            cited_chunk_ids = [
                alias_maps.chunk_alias_to_id[alias]
                for alias in raw_chunk_aliases
                if alias in alias_maps.chunk_alias_to_id
            ]
            invalid_cids = [
                cid for cid in raw_chunk_aliases
                if cid not in alias_maps.chunk_alias_to_id
            ]
            if invalid_cids:
                logger.warning(
                    "Analysis agent cited unknown chunk refs %s in section '%s'. "
                    "These references will be stripped. In full errgen runs the "
                    "checker may later flag unsupported claims; in baseline runs "
                    "they remain as unverified output.",
                    invalid_cids, section_name,
                )

            raw_calc_aliases = unique_preserve_order(
                list(item.get("calc_ids", [])) + text_calc_aliases
            )
            cited_calc_ids = [
                alias_maps.calc_alias_to_id[alias]
                for alias in raw_calc_aliases
                if alias in alias_maps.calc_alias_to_id
            ]

            # Build Citation objects for each valid chunk reference
            citations: list[Citation] = []
            for cid in cited_chunk_ids:
                chunk = chunk_map.get(cid)
                if chunk:
                    citations.append(
                        Citation(
                            chunk_id=cid,
                            source_id=chunk.source_id,
                            source_type=chunk.source_type,
                            text_snippet=chunk.text[:300],
                        )
                    )

            paragraphs.append(
                AnalysisParagraph(
                    section_name=section_name,
                    text=cleaned_text,
                    citations=citations,
                    chunk_ids=cited_chunk_ids,
                    calc_ids=cited_calc_ids,
                    verification_status=VerificationStatus.FAIL,
                )
            )

        return paragraphs
