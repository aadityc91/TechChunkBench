"""Strategy 1: Fixed-Size Token Chunking.

Splits text into chunks of exactly target_size tokens with no overlap.
Chunks may break mid-sentence or mid-word — this is intentional as a baseline.
"""

from typing import List

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        tokens = self.text_to_tokens(text)
        chunks = []
        for i in range(0, len(tokens), self.target_size):
            chunk_tokens = tokens[i : i + self.target_size]
            chunk_text = self.tokens_to_text(chunk_tokens)
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=len(chunks),
                        strategy="fixed_size",
                        metadata={
                            "doc_id": doc_id,
                            "start_token": i,
                            "end_token": i + len(chunk_tokens),
                            "target_size": self.target_size,
                        },
                    )
                )
        return chunks
