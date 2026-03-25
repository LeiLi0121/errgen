from errgen.analysis.base import BaseAnalysisAgent
from errgen.models import EvidenceChunk, SourceType
from errgen.prompt_aliases import (
    aliases_for_ids,
    build_prompt_alias_maps,
    extract_aliases_from_text,
    ids_for_aliases,
    strip_inline_aliases,
)


def test_prompt_alias_roundtrip_and_parse_response():
    chunks = [
        EvidenceChunk(
            chunk_id="real-chunk-1",
            source_id="src-1",
            source_type=SourceType.NEWS,
            text="Alpha",
        ),
        EvidenceChunk(
            chunk_id="real-chunk-2",
            source_id="src-2",
            source_type=SourceType.NEWS,
            text="Beta",
        ),
    ]
    alias_maps = build_prompt_alias_maps(chunks, [])

    assert aliases_for_ids(["real-chunk-1"], alias_maps.chunk_id_to_alias) == ["C001"]
    assert ids_for_aliases(["C002"], alias_maps.chunk_alias_to_id) == ["real-chunk-2"]

    paragraphs = BaseAnalysisAgent._parse_response(
        raw={
            "paragraphs": [
                {
                    "text": "Supported paragraph.",
                    "chunk_ids": ["C001", "BADREF"],
                    "calc_ids": [],
                }
            ]
        },
        section_name="Recent Developments",
        alias_maps=alias_maps,
        chunk_map={chunk.chunk_id: chunk for chunk in chunks},
    )

    assert len(paragraphs) == 1
    assert paragraphs[0].chunk_ids == ["real-chunk-1"]
    assert paragraphs[0].citations[0].chunk_id == "real-chunk-1"


def test_parse_response_recovers_ids_from_text_and_strips_inline_aliases():
    chunks = [
        EvidenceChunk(
            chunk_id="real-chunk-1",
            source_id="src-1",
            source_type=SourceType.NEWS,
            text="Alpha",
        ),
    ]
    alias_maps = build_prompt_alias_maps(chunks, [])

    paragraphs = BaseAnalysisAgent._parse_response(
        raw={
            "paragraphs": [
                {
                    "text": "Supported paragraph (C001).",
                    "chunk_ids": [],
                    "calc_ids": [],
                }
            ]
        },
        section_name="Recent Developments",
        alias_maps=alias_maps,
        chunk_map={chunk.chunk_id: chunk for chunk in chunks},
    )

    assert paragraphs[0].text == "Supported paragraph."
    assert paragraphs[0].chunk_ids == ["real-chunk-1"]


def test_alias_helpers_extract_and_strip_inline_references():
    chunk_aliases, calc_aliases = extract_aliases_from_text(
        "Revenue improved (C001, C002) while margin hit K003 and K004."
    )
    assert chunk_aliases == ["C001", "C002"]
    assert calc_aliases == ["K003", "K004"]
    assert strip_inline_aliases(
        "Revenue improved (C001, C002) while margin hit K003 and K004."
    ) == "Revenue improved while margin hit and."
