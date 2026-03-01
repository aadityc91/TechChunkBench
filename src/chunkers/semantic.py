"""Strategy 5: Semantic Chunking.

Splits text into sentences, embeds each, and finds breakpoints where
cosine similarity between consecutive sentences drops below a threshold.
"""

from typing import List, Optional, Set

import numpy as np

from .base import BaseChunker, Chunk, paragraph_aware_sentences, join_preserving_paragraphs


class SemanticChunker(BaseChunker):
    def __init__(self, target_size: int = 512, embed_fn=None):
        """
        Args:
            target_size: approximate target chunk size in tokens.
            embed_fn: callable that takes List[str] and returns np.ndarray of embeddings.
                      Must be provided before calling chunk().
        """
        super().__init__(target_size)
        self.embed_fn = embed_fn

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if self.embed_fn is None:
            raise ValueError("SemanticChunker requires an embed_fn. Pass it via constructor.")

        sentences, para_ends = paragraph_aware_sentences(text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text,
                    chunk_id=0,
                    strategy="semantic",
                    metadata={"doc_id": doc_id, "target_size": self.target_size},
                )
            ]

        # Embed all sentences
        embeddings = self.embed_fn(sentences)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            similarities.append(sim)

        # Adaptive threshold: mean - 1.0 * std
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        if std_sim == 0:
            # All similarities identical — no natural breakpoints;
            # let size-based post-processing handle splitting.
            threshold = -1.0
        else:
            threshold = mean_sim - 1.0 * std_sim

        # Find breakpoints
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        # Build initial chunks from breakpoints — track (sent_start_idx, group)
        pending = []
        for i in range(len(breakpoints) - 1):
            group = sentences[breakpoints[i] : breakpoints[i + 1]]
            pending.append((breakpoints[i], group))

        # Helper to join a group with paragraph boundaries
        def _join_group(group, start_idx):
            local_ends = {j - start_idx for j in para_ends
                          if start_idx <= j < start_idx + len(group)}
            return join_preserving_paragraphs(group, local_ends)

        # Post-process: iteratively split oversized chunks, merge tiny chunks
        final_groups = []  # list of (start_idx, group)
        while pending:
            sent_start, group = pending.pop(0)
            group_text = _join_group(group, sent_start)
            group_tokens = self.count_tokens(group_text)

            if group_tokens > 2 * self.target_size and len(group) > 1:
                # Similarities between consecutive sentences in this group
                sim_start = sent_start
                sim_end = sent_start + len(group) - 1
                sub_sims = similarities[sim_start:sim_end] if sim_start < sim_end else []

                if sub_sims:
                    split_at = np.argmin(sub_sims) + 1
                    # Re-queue both halves with correct sentence offsets
                    pending.insert(0, (sent_start, group[:split_at]))
                    pending.insert(1, (sent_start + split_at, group[split_at:]))
                else:
                    final_groups.append((sent_start, group))
            else:
                final_groups.append((sent_start, group))

        # Merge tiny chunks
        merged_groups = []  # list of (start_idx, group)
        i = 0
        while i < len(final_groups):
            start_idx, group = final_groups[i]
            group_text = _join_group(group, start_idx)
            group_tokens = self.count_tokens(group_text)

            if group_tokens < self.target_size // 4 and i + 1 < len(final_groups):
                # Merge with next group
                next_start, next_group = final_groups[i + 1]
                merged_groups.append((start_idx, group + next_group))
                i += 2
            else:
                merged_groups.append((start_idx, group))
                i += 1

        # If last chunk is tiny, merge with previous
        if len(merged_groups) >= 2:
            last_start, last_group = merged_groups[-1]
            last_text = _join_group(last_group, last_start)
            if self.count_tokens(last_text) < self.target_size // 4:
                prev_start, prev_group = merged_groups[-2]
                merged_groups[-2] = (prev_start, prev_group + last_group)
                merged_groups.pop()

        # Build Chunk objects
        chunks = []
        for start_idx, group in merged_groups:
            chunk_text = _join_group(group, start_idx)
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=len(chunks),
                        strategy="semantic",
                        metadata={
                            "doc_id": doc_id,
                            "num_sentences": len(group),
                            "target_size": self.target_size,
                            "similarity_threshold": float(threshold),
                        },
                    )
                )

        return chunks
