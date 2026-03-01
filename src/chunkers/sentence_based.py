"""Strategy 3: Sentence-Based Chunking.

Uses NLTK sent_tokenize to split text into sentences, then groups consecutive
sentences until reaching target_size tokens. Never breaks mid-sentence.
"""

from typing import List

from .base import BaseChunker, Chunk, paragraph_aware_sentences, join_preserving_paragraphs


class SentenceBasedChunker(BaseChunker):
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        sentences, para_ends = paragraph_aware_sentences(text)
        chunks = []
        current_sentences = []
        current_tokens = 0
        sent_start = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.target_size and current_sentences:
                # Flush current group
                local_para_ends = {j - sent_start for j in para_ends
                                   if sent_start <= j < sent_start + len(current_sentences)}
                chunk_text = join_preserving_paragraphs(current_sentences, local_para_ends)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=len(chunks),
                        strategy="sentence_based",
                        metadata={
                            "doc_id": doc_id,
                            "num_sentences": len(current_sentences),
                            "target_size": self.target_size,
                        },
                    )
                )
                sent_start = i
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens = self.count_tokens(" ".join(current_sentences))

        # Flush remaining
        if current_sentences:
            local_para_ends = {j - sent_start for j in para_ends
                               if sent_start <= j < sent_start + len(current_sentences)}
            chunk_text = join_preserving_paragraphs(current_sentences, local_para_ends)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=len(chunks),
                    strategy="sentence_based",
                    metadata={
                        "doc_id": doc_id,
                        "num_sentences": len(current_sentences),
                        "target_size": self.target_size,
                    },
                )
            )

        return chunks
