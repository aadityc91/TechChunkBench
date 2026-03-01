import tiktoken

def _get_enc():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_get_enc().encode(text))

def _truncate_at_section_boundary(text: str, max_tokens: int) -> str:
    enc = _get_enc()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated = enc.decode(tokens[:max_tokens])
    candidates = [
        truncated.rfind("\n# "),
        truncated.rfind("\n## "),
        truncated.rfind("\n### "),
    ]
    for prefix in ("# ", "## ", "### "):
        if truncated.startswith(prefix):
            candidates.append(0)
            break
            
    last_heading = max(candidates)
    
    print(f"Truncated string length: {len(truncated)}")
    print(f"Last heading index: {last_heading}")
    
    if last_heading > len(truncated) // 2:
        # STRING SLICING HERE
        truncated = truncated[:last_heading]

    return truncated.strip()

def test_truncation():
    print("--- Testing Document Truncation ---")
    
    text = "# Introduction\n\n" + ("word " * 100) + "\n\n## Section 2\n\n" + ("word " * 100)
    
    print(f"Original text tokens: {count_tokens(text)}")
    
    max_tokens = 50
    truncated_text = _truncate_at_section_boundary(text, max_tokens)
    
    print(f"Truncated text tokens: {count_tokens(truncated_text)}")
    print("Truncated text:")
    print(truncated_text)

if __name__ == "__main__":
    test_truncation()
