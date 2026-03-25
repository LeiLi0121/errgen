"""
Helpers for prompt-facing short aliases.

LLMs are much more reliable at copying short references like ``C001`` and
``K003`` than full UUIDs. Internally, ERRGen still keeps the original IDs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from errgen.models import CalculationResult, EvidenceChunk


@dataclass(frozen=True)
class PromptAliasMaps:
    chunk_alias_to_id: dict[str, str]
    chunk_id_to_alias: dict[str, str]
    calc_alias_to_id: dict[str, str]
    calc_id_to_alias: dict[str, str]


_ALIAS_RE = re.compile(r"\b([CK]\d{3})\b")
_PAREN_ALIAS_GROUP_RE = re.compile(
    r"\(\s*(?:[CK]\d{3})(?:\s*,\s*[CK]\d{3})*\s*\)"
)


def build_prompt_alias_maps(
    chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> PromptAliasMaps:
    chunk_alias_to_id = {
        f"C{i:03d}": chunk.chunk_id
        for i, chunk in enumerate(chunks, start=1)
    }
    calc_alias_to_id = {
        f"K{i:03d}": calc.calc_id
        for i, calc in enumerate(calc_results, start=1)
    }
    return PromptAliasMaps(
        chunk_alias_to_id=chunk_alias_to_id,
        chunk_id_to_alias={value: key for key, value in chunk_alias_to_id.items()},
        calc_alias_to_id=calc_alias_to_id,
        calc_id_to_alias={value: key for key, value in calc_alias_to_id.items()},
    )


def aliases_for_ids(ids: list[str], id_to_alias: dict[str, str]) -> list[str]:
    return [id_to_alias[item_id] for item_id in ids if item_id in id_to_alias]


def ids_for_aliases(aliases: list[str], alias_to_id: dict[str, str]) -> list[str]:
    return [alias_to_id[alias] for alias in aliases if alias in alias_to_id]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_aliases_from_text(text: str) -> tuple[list[str], list[str]]:
    chunk_aliases: list[str] = []
    calc_aliases: list[str] = []
    seen_chunks: set[str] = set()
    seen_calcs: set[str] = set()

    for alias in _ALIAS_RE.findall(text):
        if alias.startswith("C"):
            if alias not in seen_chunks:
                seen_chunks.add(alias)
                chunk_aliases.append(alias)
        elif alias not in seen_calcs:
            seen_calcs.add(alias)
            calc_aliases.append(alias)

    return chunk_aliases, calc_aliases


def strip_inline_aliases(text: str) -> str:
    cleaned = _PAREN_ALIAS_GROUP_RE.sub("", text)
    cleaned = _ALIAS_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
