from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import nltk
import tiktoken

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


@dataclass
class Chunk:
    text: str
    chunk_id: int
    strategy: str
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    def __init__(self, target_size: int = 512):
        """target_size: approximate target chunk size in tokens."""
        self.target_size = target_size

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        pass

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        return len(_get_encoding().encode(text))

    @staticmethod
    def tokens_to_text(tokens: list) -> str:
        """Decode a list of token ids back to text."""
        return _get_encoding().decode(tokens)

    @staticmethod
    def text_to_tokens(text: str) -> list:
        """Encode text to a list of token ids."""
        return _get_encoding().encode(text)


def paragraph_aware_sentences(text: str) -> Tuple[List[str], Set[int]]:
    """Split text into sentences while tracking paragraph boundaries.

    Returns:
        sentences: list of sentence strings
        para_ends: set of sentence indices that end a paragraph
    """
    paragraphs = text.split("\n\n")
    sentences: List[str] = []
    para_ends: Set[int] = set()
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_sents = nltk.sent_tokenize(para)
        sentences.extend(para_sents)
        if sentences:
            para_ends.add(len(sentences) - 1)
    return sentences, para_ends


def join_preserving_paragraphs(sentences: list, para_ends: Set[int]) -> str:
    """Join sentences, using \\n\\n at paragraph boundaries, space otherwise."""
    parts = []
    for i, sent in enumerate(sentences):
        parts.append(sent)
        if i < len(sentences) - 1:
            if i in para_ends:
                parts.append("\n\n")
            else:
                parts.append(" ")
    return "".join(parts)
