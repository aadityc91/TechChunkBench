"""Structure-aware variants added for the CCIS extension.

StructureLegalChunker
    The legal contracts in LegalBench-RAG carry no Markdown headings, so the
    stock StructureAwareChunker silently falls back to recursive splitting on
    them. This variant first promotes legal section markers (ARTICLE II,
    Section 5.3 Title., 1. DEFINITIONS., all-caps title lines) to Markdown
    headings, then applies the unchanged structure-aware algorithm. The
    original document text is never altered inside a chunk: only heading lines
    are lifted out into the heading path, so chunk cores still locate exactly in
    the source document for span-based evaluation.

StructureNoPrefixChunker / StructureLegalNoPrefixChunker
    Identical boundaries, but no heading-path prefix is prepended to the chunk
    text. Used for the 2x2 boundary-vs-metadata ablation.
"""
import re
from typing import List, Tuple

from .base import Chunk
from .structure_aware import StructureAwareChunker

_ROMAN = r"(?:[IVXLC]+|\d+)"
# (regex, heading level). All anchored at line start; evaluated per line.
_MARKERS: List[Tuple[re.Pattern, int]] = [
    (re.compile(rf"^\s*(ARTICLE|Article)\s+{_ROMAN}\b"), 1),
    (re.compile(r"^\s*(SECTION|Section)\s+\d+(?:\.\d+){1,3}\b"), 2),
    (re.compile(r"^\s*(SECTION|Section)\s+\d+\b"), 1),
    (re.compile(r"^\s*\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z][^\n]{1,80}$"), 2),   # 5.3 Title
    (re.compile(r"^\s*\d{1,2}\.\s+[A-Z][A-Za-z][^\n]{1,80}$"), 1),           # 1. DEFINITIONS.
    (re.compile(r"^\s*[A-Z][A-Z0-9 ,&'\-/()]{5,70}\.?\s*$"), 1),             # RECITALS / all-caps title line
]
_TITLE_END = re.compile(r"^(.{3,120}?[.:])\s+(.*)$")   # "Section 5.3 Title. body..." -> split


def promote_legal_headings(text: str, max_heading_len: int = 150) -> Tuple[str, int]:
    """Insert Markdown heading markers before legal section markers.

    Returns (new_text, n_headings). Lines that are entirely a heading become
    '#'-prefixed lines. Long lines that start with a marker and carry a short
    title ending in '.' or ':' are split into a heading line plus the untouched
    remainder, so chunk content remains verbatim document text.
    """
    out: List[str] = []
    n = 0
    for line in text.split("\n"):
        stripped = line.strip()
        level = 0
        for rx, lv in _MARKERS:
            if rx.match(line):
                level = lv
                break
        if not level or not stripped:
            out.append(line)
            continue
        if len(stripped) <= max_heading_len:
            out.append(f"{'#' * level} {stripped}")
            n += 1
            continue
        m = _TITLE_END.match(stripped)
        if m and len(m.group(1)) <= max_heading_len:
            out.append(f"{'#' * level} {m.group(1)}")
            out.append(m.group(2))
            n += 1
        else:
            out.append(line)
    return "\n".join(out), n


class StructureLegalChunker(StructureAwareChunker):
    STRATEGY = "structure_legal"

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        promoted, n = promote_legal_headings(text)
        chunks = super().chunk(promoted, doc_id)
        for c in chunks:
            c.strategy = self.STRATEGY
            c.metadata["n_promoted_headings"] = n
        return chunks


class StructureNoPrefixChunker(StructureAwareChunker):
    """Structure-aware boundaries without the '[A > B] ' heading prefix."""
    STRATEGY = "structure_noprefix"

    @staticmethod
    def _make_prefix(heading_path: List[str]) -> str:
        return ""

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        chunks = super().chunk(text, doc_id)
        for c in chunks:
            c.strategy = self.STRATEGY
        return chunks


class StructureLegalNoPrefixChunker(StructureLegalChunker):
    STRATEGY = "structure_legal_noprefix"

    @staticmethod
    def _make_prefix(heading_path: List[str]) -> str:
        return ""
