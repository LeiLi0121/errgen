"""
Helpers for prompt-facing short aliases.

LLMs are much more reliable at copying short references like ``C001`` and
``K003`` than full UUIDs. Internally, ERRGen still keeps the original IDs.
"""

from __future__ import annotations

from dataclasses import dataclass

from errgen.models import CalculationResult, EvidenceChunk


@dataclass(frozen=True)
class PromptAliasMaps:
    chunk_alias_to_id: dict[str, str]
    chunk_id_to_alias: dict[str, str]
    calc_alias_to_id: dict[str, str]
    calc_id_to_alias: dict[str, str]


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
