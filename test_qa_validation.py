import re

def test_validation_logic():
    print("--- Testing QA Validation Logic ---")
    
    # Bug 1: Unescaped regex in template generation
    # re.match(r"(.+?)\s+is defined as\s+(.+)", sent, re.IGNORECASE)
    # If the sentence has special regex chars, no problem for match.
    
    # Bug 2: The sliding window logic
    evidence = "This is a test of the emergency broadcast system."
    source_text = "Before. This is a test of the emergency broadcast system. After."
    
    import nltk
    # Assume nltk works normally
    source_sents = ["Before.", "This is a test of the emergency broadcast system.", "After."]
    
    evidence_words = set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", evidence.lower()))
    
    window_size = max(1, len(evidence.split()) // 10 + 1)
    print(f"Evidence Length: {len(evidence.split())}, Window Size: {window_size}")
    
    # What if the evidence spans multiple sentences?
    evidence_long = "Sentence one. Sentence two. Sentence three."
    print(f"Long Evidence Window Size: {max(1, len(evidence_long.split()) // 10 + 1)}")
    # 6 words // 10 = 0 + 1 = 1 window size.
    # But it spans 3 sentences!
    # So the sliding window of size 1 will NEVER contain all evidence words!
    print("If evidence is 3 short sentences (6 words), window is 1. Validation fails!")

if __name__ == "__main__":
    test_validation_logic()
