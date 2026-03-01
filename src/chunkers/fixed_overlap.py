"""Strategy 2: Fixed-Size with Overlap.

Same as fixed-size but with 20% overlap between consecutive chunks.
"""

from typing import List

from .base import BaseChunker, Chunk


class FixedOverlapChunker(BaseChunker):
    def __init__(self, target_size: int = 512, overlap_ratio: float = 0.2):
        super().__init__(target_size)
        if overlap_ratio >= 1.0:
            raise ValueError(f"overlap_ratio must be < 1.0, got {overlap_ratio}")
        self.overlap = int(target_size * overlap_ratio)
        self.step = max(1, target_size - self.overlap)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        tokens = self.text_to_tokens(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.target_size]
            chunk_text = self.tokens_to_text(chunk_tokens)
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=len(chunks),
                        strategy="fixed_overlap",
                        metadata={
                            "doc_id": doc_id,
                            "start_token": i,
                            "end_token": i + len(chunk_tokens),
                            "overlap": self.overlap,
                            "target_size": self.target_size,
                        },
                    )
                )
            i += self.step
            if i + self.target_size > len(tokens) and i < len(tokens):
                # Last partial chunk
                chunk_tokens = tokens[i:]
                chunk_text = self.tokens_to_text(chunk_tokens)
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=len(chunks),
                            strategy="fixed_overlap",
                            metadata={
                                "doc_id": doc_id,
                                "start_token": i,
                                "end_token": len(tokens),
                                "overlap": self.overlap,
                                "target_size": self.target_size,
                            },
                        )
                    )
                break
        return chunks
