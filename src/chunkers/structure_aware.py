"""Strategy 6: Structure-Aware Chunking.

Parses Markdown-style headings to build a section tree, then chunks by section.
Prepends the heading path to each chunk for context.
Falls back to recursive splitting for documents without headings.
"""

import re
from typing import List, Optional, Tuple

import nltk

from .base import BaseChunker, Chunk, paragraph_aware_sentences, join_preserving_paragraphs


class Section:
    """A section in a document tree."""

    def __init__(self, heading: str, level: int, content: str = ""):
        self.heading = heading
        self.level = level
        self.content = content
        self.children: List["Section"] = []

    def full_text(self) -> str:
        """Get the full text of this section (heading + content + children)."""
        parts = []
        if self.content.strip():
            parts.append(self.content.strip())
        for child in self.children:
            parts.append(child.full_text())
        return "\n".join(parts)


def parse_sections(text: str) -> List[Section]:
    """Parse Markdown-style headings into a flat list of sections."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections = []
    last_end = 0

    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings found — return single section
        return [Section(heading="", level=0, content=text)]

    # Content before first heading
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(Section(heading="Preamble", level=0, content=preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append(Section(heading=heading, level=level, content=content))

    return sections


def build_section_tree(flat_sections: List[Section]) -> List[Section]:
    """Build a tree from flat sections based on heading levels."""
    if not flat_sections:
        return []

    root_sections = []
    stack: List[Section] = []

    for section in flat_sections:
        # Pop stack until we find a parent with lower level.
        # Never let a level-0 node (Preamble) trap real headings.
        while stack and (stack[-1].level >= section.level or
                         (stack[-1].level == 0 and section.level > 0)):
            stack.pop()

        if stack:
            stack[-1].children.append(section)
        else:
            root_sections.append(section)

        stack.append(section)

    return root_sections


class StructureAwareChunker(BaseChunker):
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        flat_sections = parse_sections(text)

        # If no real headings found, fall back to recursive splitting
        if len(flat_sections) == 1 and flat_sections[0].level == 0 and flat_sections[0].heading in ("", "Preamble"):
            from .recursive import RecursiveChunker
            fallback = RecursiveChunker(target_size=self.target_size)
            chunks = fallback.chunk(text, doc_id)
            for c in chunks:
                c.strategy = "structure_aware"
            return chunks

        tree = build_section_tree(flat_sections)
        chunks = []
        self._process_sections(tree, [], chunks, doc_id)

        # Post-process: merge tiny chunks
        merged = self._merge_tiny_chunks(chunks)
        # Re-index
        for i, c in enumerate(merged):
            c.chunk_id = i

        return merged

    def _process_sections(
        self,
        sections: List[Section],
        heading_path: List[str],
        chunks: List[Chunk],
        doc_id: str,
    ):
        for section in sections:
            current_path = heading_path + [section.heading] if section.heading else heading_path
            section_text = section.content.strip() if section.content else ""
            section_tokens = self.count_tokens(section_text) if section_text else 0

            if section.children:
                # If this section has direct content before children, add it
                if section_text and section_tokens > 0:
                    if section_tokens > self.target_size:
                        # Split large content with sentence-based approach
                        sub_chunks = self._sentence_split(section_text, current_path, doc_id)
                        chunks.extend(sub_chunks)
                    else:
                        prefix = self._make_prefix(current_path)
                        chunks.append(
                            Chunk(
                                text=f"{prefix}{section_text}",
                                chunk_id=len(chunks),
                                strategy="structure_aware",
                                metadata={
                                    "doc_id": doc_id,
                                    "heading_path": " > ".join(current_path),
                                    "target_size": self.target_size,
                                },
                            )
                        )
                # Recurse into children
                self._process_sections(section.children, current_path, chunks, doc_id)
            else:
                # Leaf section
                if not section_text:
                    continue

                if section_tokens > self.target_size:
                    # Split with sentence-based approach
                    sub_chunks = self._sentence_split(section_text, current_path, doc_id)
                    chunks.extend(sub_chunks)
                else:
                    prefix = self._make_prefix(current_path)
                    chunks.append(
                        Chunk(
                            text=f"{prefix}{section_text}",
                            chunk_id=len(chunks),
                            strategy="structure_aware",
                            metadata={
                                "doc_id": doc_id,
                                "heading_path": " > ".join(current_path),
                                "target_size": self.target_size,
                            },
                        )
                    )

    def _sentence_split(self, text: str, heading_path: List[str], doc_id: str) -> List[Chunk]:
        """Split oversized section content by sentences."""
        sentences, para_ends = paragraph_aware_sentences(text)
        prefix = self._make_prefix(heading_path)
        prefix_tokens = self.count_tokens(prefix) if prefix else 0
        effective_limit = max(self.target_size - prefix_tokens, 64)
        chunks = []
        current = []
        current_tokens = 0
        sent_start = 0

        for i, sent in enumerate(sentences):
            sent_tokens = self.count_tokens(sent)
            if current_tokens + sent_tokens > effective_limit and current:
                local_para_ends = {j - sent_start for j in para_ends
                                   if sent_start <= j < sent_start + len(current)}
                chunk_text = prefix + join_preserving_paragraphs(current, local_para_ends)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=0,
                        strategy="structure_aware",
                        metadata={
                            "doc_id": doc_id,
                            "heading_path": " > ".join(heading_path),
                            "target_size": self.target_size,
                        },
                    )
                )
                sent_start = i
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens = self.count_tokens(" ".join(current))

        if current:
            local_para_ends = {j - sent_start for j in para_ends
                               if sent_start <= j < sent_start + len(current)}
            chunk_text = prefix + join_preserving_paragraphs(current, local_para_ends)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=0,
                    strategy="structure_aware",
                    metadata={
                        "doc_id": doc_id,
                        "heading_path": " > ".join(heading_path),
                        "target_size": self.target_size,
                    },
                )
            )

        return chunks

    def _merge_tiny_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than target_size / 4 with adjacent chunks from the same section."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            c_tokens = self.count_tokens(c.text)
            if c_tokens < self.target_size // 4 and i + 1 < len(chunks):
                next_c = chunks[i + 1]
                # Only merge if chunks belong to the same section
                if c.metadata.get("heading_path") == next_c.metadata.get("heading_path"):
                    merged_text = c.text + "\n\n" + next_c.text
                    merged.append(
                        Chunk(
                            text=merged_text,
                            chunk_id=0,
                            strategy="structure_aware",
                            metadata=next_c.metadata,
                        )
                    )
                    i += 2
                else:
                    merged.append(c)
                    i += 1
            else:
                merged.append(c)
                i += 1

        # If last chunk is tiny, merge with previous chunk (same section only)
        if len(merged) >= 2:
            last_tokens = self.count_tokens(merged[-1].text)
            if last_tokens < self.target_size // 4:
                if merged[-2].metadata.get("heading_path") == merged[-1].metadata.get("heading_path"):
                    prev = merged[-2]
                    merged_text = prev.text + "\n\n" + merged[-1].text
                    merged[-2] = Chunk(
                        text=merged_text,
                        chunk_id=0,
                        strategy="structure_aware",
                        metadata=prev.metadata,
                    )
                    merged.pop()

        return merged

    @staticmethod
    def _make_prefix(heading_path: List[str]) -> str:
        if not heading_path:
            return ""
        return f"[{' > '.join(heading_path)}] "
