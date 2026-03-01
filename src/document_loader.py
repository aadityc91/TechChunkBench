"""Document loading and processing utilities."""

import json
import logging
import os
import re

import tiktoken

from config import CORPORA, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

_ENC = None


def _get_enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    return len(_get_enc().encode(text))


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML, preserving heading structure as Markdown."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove nav, footer, sidebar, script, style
    for tag in soup.find_all(["nav", "footer", "aside", "script", "style", "header"]):
        tag.decompose()

    # Remove elements with common navigation class/id names
    for selector in [".sidebar", ".nav", ".footer", ".header", ".breadcrumb",
                     "#sidebar", "#nav", "#footer", "#header", "#breadcrumb",
                     "[role='navigation']", "[role='banner']", "[role='contentinfo']"]:
        for el in soup.select(selector):
            el.decompose()

    # Convert to markdown (preserves headings)
    md = markdownify(str(soup), heading_style="ATX", strip=["img", "a"])

    # Clean up excessive whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip()

    return md


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF, detecting headings heuristically."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text.append(page_text)

    text = "\n".join(full_text)

    # Heuristic heading detection
    lines = text.split("\n")
    processed = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            processed.append("")
            continue

        # Detect headings: ALL CAPS lines, short lines followed by blank
        is_heading = False
        heading_level = "#"

        # Pattern: ALL CAPS, reasonable length
        if stripped.isupper() and 3 < len(stripped) < 100 and not stripped.startswith("("):
            is_heading = True
            heading_level = "#"

        # Pattern: "Section X.Y" or "Chapter X" style
        if re.match(r"^(Section|Chapter|Part|Article|Appendix)\s+[\dA-Z]", stripped, re.IGNORECASE):
            is_heading = True
            heading_level = "##"

        # Pattern: "X.Y Title" or "X.Y.Z Title" (numbered sections)
        if re.match(r"^\d+\.\d+(\.\d+)?\s+[A-Z]", stripped):
            is_heading = True
            heading_level = "##"
        elif (re.match(r"^\d+\s+[A-Z][a-z]", stripped)
              and len(stripped) < 80
              and len(stripped.split()) <= 8
              and not re.match(r"^\d+\s+(GB|MB|TB|KB|GHz|MHz|mm|cm|kg|ms)\b", stripped)):
            is_heading = True
            heading_level = "#"

        if is_heading:
            processed.append(f"{heading_level} {stripped}")
        else:
            processed.append(stripped)

    result = "\n".join(processed)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def save_processed_document(corpus_id: str, text: str, metadata: dict):
    """Save processed document text and metadata."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Enforce token limits
    total_tokens = count_tokens(text)
    if total_tokens > 80000:
        # Truncate at section boundary
        text = _truncate_at_section_boundary(text, 80000)
        total_tokens = count_tokens(text)

    # Save text
    text_path = os.path.join(PROCESSED_DIR, f"{corpus_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Update and save metadata
    metadata["total_tokens"] = total_tokens
    metadata["total_sections"] = text.count("\n# ") + text.count("\n## ") + text.count("\n### ")
    meta_path = os.path.join(PROCESSED_DIR, f"{corpus_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {corpus_id}: {total_tokens} tokens, {metadata['total_sections']} sections")


def _truncate_at_section_boundary(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens at the nearest section boundary."""
    enc = _get_enc()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated = enc.decode(tokens[:max_tokens])
    # Find last heading (check both \n-prefixed and start-of-string)
    candidates = [
        truncated.rfind("\n# "),
        truncated.rfind("\n## "),
        truncated.rfind("\n### "),
    ]
    # Also check if the document starts with a heading
    for prefix in ("# ", "## ", "### "):
        if truncated.startswith(prefix):
            candidates.append(0)
            break
    last_heading = max(candidates)
    if last_heading > len(truncated) // 2:
        truncated = truncated[:last_heading]

    return truncated.strip()


def load_document(corpus_id: str) -> str:
    """Load a processed document by corpus ID."""
    path = os.path.join(PROCESSED_DIR, f"{corpus_id}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_corpora() -> dict:
    """Load all processed documents defined in CORPORA config. Returns {corpus_id: text}."""
    corpora = {}
    for corpus_id in CORPORA:
        path = os.path.join(PROCESSED_DIR, f"{corpus_id}.txt")
        if os.path.exists(path):
            corpora[corpus_id] = load_document(corpus_id)
    return corpora


def load_metadata(corpus_id: str) -> dict:
    """Load metadata for a corpus."""
    path = os.path.join(PROCESSED_DIR, f"{corpus_id}_meta.json")
    with open(path) as f:
        return json.load(f)
