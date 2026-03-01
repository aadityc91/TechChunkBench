"""Strategy 7: Hybrid (Structure + Semantic) Chunking.

First applies structure-aware chunking, then uses semantic splitting for
oversized chunks without sub-headings, and semantic merging for tiny chunks.
"""

import logging
from typing import List, Optional

import numpy as np

from .base import BaseChunker, Chunk
from .structure_aware import StructureAwareChunker
from .semantic import SemanticChunker

logger = logging.getLogger(__name__)


class HybridChunker(BaseChunker):
    def __init__(self, target_size: int = 512, embed_fn=None):
        """
        Args:
            target_size: approximate target chunk size in tokens.
            embed_fn: callable that takes List[str] and returns np.ndarray of embeddings.
                      Required for semantic splitting/merging of oversized/tiny chunks.
        """
        super().__init__(target_size)
        self.embed_fn = embed_fn
        self.structure_chunker = StructureAwareChunker(target_size=target_size)
        if embed_fn is None:
            logger.warning(
                "HybridChunker created without embed_fn. Semantic splitting/merging "
                "will be skipped, effectively falling back to structure_aware chunking."
            )

    @staticmethod
    def _extract_prefix(text: str):
        """Extract heading prefix like '[A > B] ' from chunk text.

        Returns (prefix, body) where prefix includes trailing space,
        or ('', text) if no prefix is found.
        """
        if text.startswith("["):
            close = text.find("] ")
            if close != -1:
                return text[: close + 2], text[close + 2 :]
        return "", text

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        # Step 1: Apply structure-aware chunking
        initial_chunks = self.structure_chunker.chunk(text, doc_id)

        # Step 2: Split oversized chunks using semantic chunking
        refined_chunks = []
        for c in initial_chunks:
            c_tokens = self.count_tokens(c.text)
            if c_tokens > self.target_size * 1.5 and self.embed_fn is not None:
                # Strip heading prefix before semantic splitting, then
                # re-prepend it to every sub-chunk so structural context
                # is preserved (mirrors structure_aware._sentence_split).
                prefix, body = self._extract_prefix(c.text)
                prefix_tokens = self.count_tokens(prefix) if prefix else 0

                semantic_chunker = SemanticChunker(
                    target_size=max(self.target_size - prefix_tokens, 64),
                    embed_fn=self.embed_fn,
                )
                sub_chunks = semantic_chunker.chunk(body, doc_id)

                for sc in sub_chunks:
                    sc.text = prefix + sc.text
                    sc.strategy = "hybrid"
                    sc.metadata.update({
                        "heading_path": c.metadata.get("heading_path", ""),
                        "split_method": "semantic",
                    })
                refined_chunks.extend(sub_chunks)
            else:
                c.strategy = "hybrid"
                refined_chunks.append(c)

        # Step 3: Merge tiny chunks using semantic similarity
        if self.embed_fn is not None:
            merged_chunks = self._semantic_merge(refined_chunks)
        else:
            merged_chunks = refined_chunks

        # Re-index
        for i, c in enumerate(merged_chunks):
            c.chunk_id = i

        return merged_chunks

    def _semantic_merge(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than target_size/4 based on semantic similarity (same section only)."""
        if len(chunks) <= 1:
            return chunks

        tiny_threshold = self.target_size // 4
        merged = []
        i = 0

        while i < len(chunks):
            c = chunks[i]
            c_tokens = self.count_tokens(c.text)

            if c_tokens < tiny_threshold and i + 1 < len(chunks):
                next_c = chunks[i + 1]
                # Only merge if chunks share the same heading_path
                if c.metadata.get("heading_path") == next_c.metadata.get("heading_path"):
                    embeddings = self.embed_fn([c.text, next_c.text])
                    sim = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10
                    )
                    if sim > 0.7:
                        merged_text = c.text + "\n\n" + next_c.text
                        merged.append(
                            Chunk(
                                text=merged_text,
                                chunk_id=0,
                                strategy="hybrid",
                                metadata={
                                    **next_c.metadata,
                                    "merged": True,
                                    "merge_similarity": float(sim),
                                },
                            )
                        )
                        i += 2
                        continue

            merged.append(c)
            i += 1

        # If last chunk is tiny, merge with previous
        if len(merged) >= 2:
            last_tokens = self.count_tokens(merged[-1].text)
            if last_tokens < tiny_threshold:
                prev = merged[-2]
                last = merged[-1]
                if prev.metadata.get("heading_path") == last.metadata.get("heading_path"):
                    merged[-2] = Chunk(
                        text=prev.text + "\n\n" + last.text,
                        chunk_id=0,
                        strategy="hybrid",
                        metadata={**prev.metadata, "merged_trailing": True},
                    )
                    merged.pop()

        return merged
