"""Strategy 4: Recursive Character Splitter.

Uses LangChain's RecursiveCharacterTextSplitter — the industry-standard approach.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    def __init__(self, target_size: int = 512):
        super().__init__(target_size)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=target_size,
            chunk_overlap=max(1, target_size // 5),
            length_function=self.count_tokens,
        )

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        splits = self.splitter.split_text(text)
        chunks = []
        for split_text in splits:
            if split_text.strip():
                chunks.append(
                    Chunk(
                        text=split_text,
                        chunk_id=len(chunks),
                        strategy="recursive",
                        metadata={
                            "doc_id": doc_id,
                            "target_size": self.target_size,
                        },
                    )
                )
        return chunks
