from .base import BaseChunker, Chunk
from .fixed_size import FixedSizeChunker
from .fixed_overlap import FixedOverlapChunker
from .sentence_based import SentenceBasedChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .structure_aware import StructureAwareChunker
from .hybrid import HybridChunker

CHUNKER_MAP = {
    "fixed_size": FixedSizeChunker,
    "fixed_overlap": FixedOverlapChunker,
    "sentence_based": SentenceBasedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "structure_aware": StructureAwareChunker,
    "hybrid": HybridChunker,
}


def get_chunker(strategy_name: str, **kwargs) -> BaseChunker:
    """Factory function to get a chunker by strategy name."""
    if strategy_name not in CHUNKER_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(CHUNKER_MAP.keys())}")
    return CHUNKER_MAP[strategy_name](**kwargs)
