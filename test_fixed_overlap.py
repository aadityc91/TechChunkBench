from src.chunkers.fixed_overlap import FixedOverlapChunker

def test_infinite_loop():
    print("--- Testing FixedOverlap Infinite Loop ---")
    try:
        chunker = FixedOverlapChunker(target_size=512, overlap_ratio=1.0) # Caught by ValueError
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        
    chunker = FixedOverlapChunker(target_size=10, overlap_ratio=0.5)
    print(f"Target Size: {chunker.target_size}, Overlap: {chunker.overlap}, Step: {chunker.step}")
    
    text = "word "*20 # 20 tokens
    chunks = chunker.chunk(text, "doc1")
    print(f"Produced {len(chunks)} chunks.")
    
    # What if overlap_ratio is effectively 1.0 due to integer truncation?
    # e.g., target_size=1, overlap_ratio=0.9
    chunker = FixedOverlapChunker(target_size=1, overlap_ratio=0.9)
    print(f"\nTarget Size: {chunker.target_size}, Overlap: {chunker.overlap}, Step: {chunker.step}")
    
    chunks = chunker.chunk("a b c", "doc1")
    print(f"Produced {len(chunks)} chunks.")

test_infinite_loop()
