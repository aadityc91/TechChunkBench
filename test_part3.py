import tiktoken
from src.chunkers.structure_aware import StructureAwareChunker
from src.chunkers.hybrid import HybridChunker

def test_negative_target_size():
    print("--- Testing Negative Target Size in StructureAware ---")
    chunker = StructureAwareChunker(target_size=10)
    # create a very long heading
    heading = "# " + "long " * 20
    content = "This is a sentence. This is another sentence. This is a third sentence."
    text = heading + "\n" + content
    
    chunks = chunker.chunk(text, "doc1")
    print(f"Produced {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c.text}")

def test_rouge_proxy():
    print("\n--- Testing ROUGE Proxy Logic ---")
    from src.evaluator import get_most_relevant_sentence, compute_rouge_l
    question = "What is the answer?"
    evidence = "The answer is 42."
    context = "This is context. The answer is 42. Some other info."
    extracted = get_most_relevant_sentence(context, question, evidence)
    rouge = compute_rouge_l(extracted, evidence)
    print(f"Extracted: '{extracted}'")
    print(f"ROUGE-L: {rouge}")

def test_truncation():
    print("\n--- Testing Document Truncation Bug ---")
    from src.document_loader import _truncate_at_section_boundary
    text = "# Start\n" + "word " * 10
    trunc = _truncate_at_section_boundary(text, 5)
    print(f"Truncated: '{trunc}'")

if __name__ == "__main__":
    test_negative_target_size()
    test_rouge_proxy()
    test_truncation()
