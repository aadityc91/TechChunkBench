"""Download and process the 24 document corpora for TechChunkBench (3 per domain)."""

import datetime
import logging
import os
import sys
import time

import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DIR, PROCESSED_DIR, CORPORA
from src.document_loader import (
    extract_text_from_html,
    extract_text_from_pdf,
    save_processed_document,
    count_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) TechChunkBench Research Bot"
}


def download_url(url: str, save_path: str, timeout: int = 60) -> bool:
    """Download a URL to a local file."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mode = "wb" if save_path.endswith(".pdf") else "w"
        encoding = None if mode == "wb" else "utf-8"
        with open(save_path, mode, encoding=encoding) as f:
            if mode == "wb":
                f.write(response.content)
            else:
                f.write(response.content.decode("utf-8", errors="replace"))
        logger.info(f"Downloaded: {url} -> {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def scrape_multipage(base_url: str, link_selector: str, save_dir: str,
                     max_pages: int = 50) -> str:
    """Scrape multiple linked pages from a documentation site and combine."""
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for a in soup.select(link_selector):
            href = a.get("href", "")
            if href and not href.startswith(("javascript:", "#", "mailto:")):
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(base_url, href)
                links.append(href)

        links = list(dict.fromkeys(links))[:max_pages]  # dedupe, limit
        logger.info(f"Found {len(links)} linked pages from {base_url}")

        combined_html = []
        for i, link in enumerate(links):
            try:
                time.sleep(0.5)  # Be polite
                resp = requests.get(link, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                combined_html.append(resp.text)
                logger.info(f"  Scraped page {i+1}/{len(links)}: {link}")
            except Exception as e:
                logger.warning(f"  Failed page {link}: {e}")

        return "\n".join(combined_html)

    except Exception as e:
        logger.error(f"Failed to scrape {base_url}: {e}")
        return ""


def _convert_asciidoc_to_markdown(text: str) -> str:
    """Convert AsciiDoc headings and common syntax to Markdown.

    Handles heading conversion (= Title → # Title), strips anchor blocks
    ([[id]]), and removes video/image embed macros so the result is
    Markdown-compatible for downstream chunkers that parse # headings.
    """
    import re
    lines = text.split("\n")
    result = []
    for line in lines:
        # Convert AsciiDoc headings: = Title, == Section, === Subsection, ...
        m = re.match(r"^(={1,6})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            result.append("#" * level + " " + m.group(2))
        # Strip [[anchor]] lines
        elif re.match(r"^\[\[.+\]\]\s*$", line):
            continue
        # Strip video/image embed macros
        elif re.match(r"^(video|image)::", line):
            continue
        else:
            result.append(line)
    return "\n".join(result)


# --- Individual corpus download functions ---


def download_gdpr():
    """Download GDPR full text from EUR-Lex."""
    corpus_id = "gdpr"
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
    raw_path = os.path.join(RAW_DIR, "gdpr.html")

    if download_url(url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        save_processed_document(corpus_id, text, {
            "corpus_id": corpus_id,
            "domain": "legal_regulatory",
            "source_url": url,
            "download_date": str(datetime.date.today()),
        })
        return True

    logger.error("GDPR download failed. No suitable fallback available — "
                 "the GDPR corpus requires the actual EU regulation text.")
    return False


def download_postgresql():
    """Download PostgreSQL 16 documentation (selected chapters)."""
    corpus_id = "postgresql"
    chapters = [
        "https://www.postgresql.org/docs/16/sql-commands.html",
        "https://www.postgresql.org/docs/16/datatype.html",
        "https://www.postgresql.org/docs/16/indexes.html",
        "https://www.postgresql.org/docs/16/sql-select.html",
        "https://www.postgresql.org/docs/16/sql-insert.html",
        "https://www.postgresql.org/docs/16/sql-update.html",
        "https://www.postgresql.org/docs/16/sql-delete.html",
        "https://www.postgresql.org/docs/16/sql-createtable.html",
        "https://www.postgresql.org/docs/16/sql-altertable.html",
        "https://www.postgresql.org/docs/16/functions.html",
    ]

    combined_html = []
    for url in chapters:
        raw_path = os.path.join(RAW_DIR, f"pg_{url.split('/')[-1]}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        full_html = "\n".join(combined_html)
        text = extract_text_from_html(full_html)
        save_processed_document(corpus_id, text, {
            "corpus_id": corpus_id,
            "domain": "software_manual",
            "source_url": "https://www.postgresql.org/docs/16/",
            "download_date": str(datetime.date.today()),
        })
        return True

    return False


def download_arduino():
    """Download Arduino documentation from multiple substantial sources."""
    corpus_id = "arduino"
    # Use FreeCodeCamp Arduino Handbook + SparkFun guides for substantial content
    sources = [
        ("https://www.freecodecamp.org/news/the-arduino-handbook/", "arduino_handbook.html"),
        ("https://learn.sparkfun.com/tutorials/sparkfun-inventors-kit-experiment-guide---v41/all", "arduino_sparkfun_v41.html"),
        ("https://learn.sparkfun.com/tutorials/what-is-an-arduino/all", "arduino_sparkfun_intro.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  Arduino source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "hardware_manual",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Last resort fallback: SparkFun SIK v3.2
    logger.info("Arduino primary sources insufficient. Trying SIK v3.2...")
    fallback_url = "https://learn.sparkfun.com/tutorials/sik-experiment-guide-for-arduino---v32/all"
    raw_path = os.path.join(RAW_DIR, "arduino_sparkfun_v32.html")
    if download_url(fallback_url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "hardware_manual",
                "source_url": fallback_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_aws_s3():
    """Download AWS S3 User Guide pages."""
    corpus_id = "aws_s3"
    # AWS docs can be tricky - try the main userguide index
    pages = [
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/uploading-downloading-objects.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/security.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-overview.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-policies.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/serv-side-encryption.html",        # Encryption
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/monitoring-overview.html",          # Monitoring
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventNotifications.html",           # Event notifications
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/transfer-acceleration.html",        # Transfer acceleration
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html",               # Static website hosting
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html",                  # Multipart upload
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html",       # Performance optimization
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html",                  # Object lock / WORM
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-points.html",                # Access points
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/batch-ops.html",                    # Batch operations
    ]

    combined_html = []
    for url in pages:
        raw_path = os.path.join(RAW_DIR, f"aws_{url.split('/')[-1]}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        save_processed_document(corpus_id, text, {
            "corpus_id": corpus_id,
            "domain": "cloud_api",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/",
            "download_date": str(datetime.date.today()),
        })
        return True

    # Fallback: Google Cloud Storage docs
    logger.info("AWS S3 docs failed. Trying GCS as fallback...")
    fallback_url = "https://cloud.google.com/storage/docs/introduction"
    raw_path = os.path.join(RAW_DIR, "gcs.html")
    if download_url(fallback_url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        save_processed_document(corpus_id, text, {
            "corpus_id": corpus_id,
            "domain": "cloud_api",
            "source_url": fallback_url,
            "download_date": str(datetime.date.today()),
        })
        return True

    return False


def download_nist():
    """Download NIST SP 800-53 Rev 5."""
    corpus_id = "nist_800_53"
    # Try the HTML version first
    url = "https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final"
    raw_path = os.path.join(RAW_DIR, "nist_800_53.html")

    if download_url(url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "technical_standard",
                "source_url": url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Try PDF version
    pdf_url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"
    pdf_path = os.path.join(RAW_DIR, "nist_800_53.pdf")
    if download_url(pdf_url, pdf_path, timeout=120):
        text = extract_text_from_pdf(pdf_path)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "technical_standard",
                "source_url": pdf_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_basel():
    """Download Basel III / financial regulation documents."""
    corpus_id = "basel_iii"
    url = "https://www.bis.org/bcbs/basel3.htm"
    raw_path = os.path.join(RAW_DIR, "basel_iii.html")

    if download_url(url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)

        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "financial_compliance",
                "source_url": url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: SEC EDGAR filing (a public 10-K)
    logger.info("Basel III failed. Trying SEC filing as fallback...")
    # Apple's 10-K is publicly available
    fallback_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    raw_path = os.path.join(RAW_DIR, "sec_10k.html")
    if download_url(fallback_url, raw_path, timeout=120):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        save_processed_document(corpus_id, text, {
            "corpus_id": corpus_id,
            "domain": "financial_compliance",
            "source_url": fallback_url,
            "download_date": str(datetime.date.today()),
        })
        return True

    return False


def download_who():
    """Download medical/malaria guidelines from CDC + NCBI sources."""
    corpus_id = "who_malaria"

    # Combine multiple CDC/NCBI sources for substantial content
    sources = [
        ("https://www.cdc.gov/yellow-book/hcp/travel-associated-infections-diseases/malaria.html", "cdc_yellowbook_malaria.html"),
        ("https://www.cdc.gov/malaria/hcp/clinical-guidance/malaria-treatment-tables.html", "cdc_treatment_tables.html"),
        ("https://www.cdc.gov/malaria/hcp/clinical-guidance/treatment-uncomplicated.html", "cdc_treatment_uncomplicated.html"),
        ("https://www.cdc.gov/malaria/hcp/clinical-guidance/treatment-of-severe-malaria.html", "cdc_treatment_severe.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK294441/", "who_ncbi_uncomplicated.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK294445/?report=printable", "who_ncbi_severe.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK551711/", "ncbi_statpearls_malaria.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK8584/", "ncbi_medsci_malaria.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  Malaria source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        total_tokens = count_tokens(full_text)
        logger.info(f"  Combined malaria corpus: {total_tokens} tokens from {len(combined_text)} sources")
        if total_tokens >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "medical_guideline",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: PMC review articles
    logger.info("CDC/NCBI sources insufficient. Trying PMC review articles...")
    pmc_urls = [
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC11442732/", "pmc_malaria_africa.html"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC11597227/", "pmc_malaria_treatment.html"),
    ]
    for url, filename in pmc_urls:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            if count_tokens(text) > 500:
                combined_text.append(text)
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        save_processed_document(corpus_id, full_text, {
            "corpus_id": corpus_id,
            "domain": "medical_guideline",
            "source_url": source_url or pmc_urls[0][0],
            "download_date": str(datetime.date.today()),
        })
        return True

    return False


def download_nasa():
    """Download NASA software engineering standards from multiple sources and combine."""
    corpus_id = "nasa_std"

    # Combine multiple NASA standards for sufficient corpus size
    html_sources = [
        ("https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7150_002C_", "nasa_npr_7150_2c.html"),
        ("https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7150_002A_&page_name=all", "nasa_npr_7150_2a.html"),
        ("https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7123_001B_&page_name=ALL", "nasa_npr_7123.html"),
    ]

    pdf_sources = [
        ("https://standards.nasa.gov/sites/default/files/standards/NASA/B/0/NASA-STD-87398-Revision-B.pdf", "nasa_std_8739_8b.pdf"),
    ]

    combined_text = []
    source_url = ""

    # Download HTML sources
    for url, filename in html_sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path, timeout=120):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  NASA source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(1)

    # Download PDF sources
    for url, filename in pdf_sources:
        pdf_path = os.path.join(RAW_DIR, filename)
        if download_url(url, pdf_path, timeout=120):
            text = extract_text_from_pdf(pdf_path)
            tokens = count_tokens(text)
            logger.info(f"  NASA source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(1)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        total_tokens = count_tokens(full_text)
        logger.info(f"  Combined NASA corpus: {total_tokens} tokens from {len(combined_text)} sources")
        if total_tokens >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "engineering_spec",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


# --- New corpora download functions (n=3 expansion) ---


def download_azure_blob():
    """Download Azure Blob Storage documentation pages."""
    corpus_id = "azure_blob"
    pages = [
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-overview",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-upload-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-download-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-delete-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-containers-list-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-list-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-copy-python",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/access-tiers-overview",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/encryption-customer-provided-keys",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-overview",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-overview",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-immutability-policies-manage",
        "https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-data-operations-portal",
    ]

    combined_html = []
    for url in pages:
        raw_path = os.path.join(RAW_DIR, f"azure_{url.split('/')[-1]}.html")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "cloud_api",
                "source_url": "https://learn.microsoft.com/en-us/azure/storage/blobs/",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_gcs():
    """Download Google Cloud Storage documentation pages."""
    corpus_id = "gcs"
    pages = [
        "https://cloud.google.com/storage/docs/introduction",
        "https://cloud.google.com/storage/docs/creating-buckets",
        "https://cloud.google.com/storage/docs/uploading-objects",
        "https://cloud.google.com/storage/docs/downloading-objects",
        "https://cloud.google.com/storage/docs/listing-objects",
        "https://cloud.google.com/storage/docs/deleting-objects",
        "https://cloud.google.com/storage/docs/access-control",
        "https://cloud.google.com/storage/docs/encryption",
        "https://cloud.google.com/storage/docs/object-versioning",
        "https://cloud.google.com/storage/docs/lifecycle",
        "https://cloud.google.com/storage/docs/storage-classes",
        "https://cloud.google.com/storage/docs/object-hold",
        "https://cloud.google.com/storage/docs/bucket-lock",
        "https://cloud.google.com/storage/docs/requester-pays",
        "https://cloud.google.com/storage/docs/json_api/v1/objects",
    ]

    combined_html = []
    for url in pages:
        raw_path = os.path.join(RAW_DIR, f"gcs_{url.split('/')[-1]}.html")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "cloud_api",
                "source_url": "https://cloud.google.com/storage/docs/",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_mysql():
    """Download MySQL 8.0 Reference Manual (selected chapters)."""
    corpus_id = "mysql"
    chapters = [
        "https://dev.mysql.com/doc/refman/8.0/en/sql-data-definition-statements.html",
        "https://dev.mysql.com/doc/refman/8.0/en/data-types.html",
        "https://dev.mysql.com/doc/refman/8.0/en/select.html",
        "https://dev.mysql.com/doc/refman/8.0/en/insert.html",
        "https://dev.mysql.com/doc/refman/8.0/en/update.html",
        "https://dev.mysql.com/doc/refman/8.0/en/delete.html",
        "https://dev.mysql.com/doc/refman/8.0/en/create-table.html",
        "https://dev.mysql.com/doc/refman/8.0/en/alter-table.html",
        "https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html",
        "https://dev.mysql.com/doc/refman/8.0/en/innodb-storage-engine.html",
    ]

    combined_html = []
    for url in chapters:
        raw_path = os.path.join(RAW_DIR, f"mysql_{url.split('/')[-1]}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "software_manual",
                "source_url": "https://dev.mysql.com/doc/refman/8.0/en/",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_sqlite():
    """Download SQLite documentation pages."""
    corpus_id = "sqlite"
    pages = [
        "https://www.sqlite.org/lang.html",
        "https://www.sqlite.org/lang_select.html",
        "https://www.sqlite.org/lang_insert.html",
        "https://www.sqlite.org/lang_update.html",
        "https://www.sqlite.org/lang_delete.html",
        "https://www.sqlite.org/lang_createtable.html",
        "https://www.sqlite.org/lang_altertable.html",
        "https://www.sqlite.org/datatype3.html",
        "https://www.sqlite.org/queryplanner.html",
        "https://www.sqlite.org/fts5.html",
        "https://www.sqlite.org/json1.html",
        "https://www.sqlite.org/wal.html",
        "https://www.sqlite.org/pragma.html",
    ]

    combined_html = []
    for url in pages:
        raw_path = os.path.join(RAW_DIR, f"sqlite_{url.split('/')[-1]}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "software_manual",
                "source_url": "https://www.sqlite.org/docs.html",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_raspberry_pi():
    """Download Raspberry Pi documentation from GitHub raw AsciiDoc sources."""
    corpus_id = "raspberry_pi"
    # The raspberrypi.com site is a JS SPA; use raw GitHub sources instead
    base = "https://raw.githubusercontent.com/raspberrypi/documentation/master/documentation/asciidoc/computers"
    sources = [
        (f"{base}/getting-started/setting-up.adoc", "rpi_setting_up.adoc"),
        (f"{base}/getting-started/configuring.adoc", "rpi_configuring.adoc"),
        (f"{base}/getting-started/install.adoc", "rpi_install.adoc"),
        (f"{base}/configuration/raspi-config.adoc", "rpi_raspi_config.adoc"),
        (f"{base}/raspberry-pi/introduction.adoc", "rpi_introduction.adoc"),
        (f"{base}/linux_kernel/building.adoc", "rpi_kernel_building.adoc"),
        (f"{base}/camera/rpicam_apps_getting_started.adoc", "rpi_camera.adoc"),
        (f"{base}/remote-access/introduction.adoc", "rpi_remote_access.adoc"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                text = _convert_asciidoc_to_markdown(f.read())
            tokens = count_tokens(text)
            logger.info(f"  RPi source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.3)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "hardware_manual",
                "source_url": "https://github.com/raspberrypi/documentation",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_esp32():
    """Download ESP-IDF Programming Guide."""
    corpus_id = "esp32"
    pages = [
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/gpio.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/uart.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/spi_master.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/i2c.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/adc_oneshot.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_wifi.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/storage/nvs_flash.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/freertos.html",
        "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/log.html",
    ]

    combined_html = []
    for url in pages:
        raw_path = os.path.join(RAW_DIR, f"esp32_{url.split('/')[-2]}_{url.split('/')[-1]}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                combined_html.append(f.read())
            time.sleep(0.5)

    if combined_html:
        text = extract_text_from_html("\n".join(combined_html))
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "hardware_manual",
                "source_url": "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_ccpa():
    """Download California Consumer Privacy Act (CCPA) text."""
    corpus_id = "ccpa"
    # California Legislative Information
    url = "https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?division=3.&part=4.&lawCode=CIV&title=1.81.5"
    raw_path = os.path.join(RAW_DIR, "ccpa.html")

    if download_url(url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "legal_regulatory",
                "source_url": url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: IAPP CCPA text
    logger.info("CCPA primary failed. Trying IAPP fallback...")
    fallback_url = "https://iapp.org/resources/article/california-consumer-privacy-act-of-2018/"
    raw_path = os.path.join(RAW_DIR, "ccpa_fallback.html")
    if download_url(fallback_url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "legal_regulatory",
                "source_url": fallback_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_hipaa():
    """Download HIPAA Privacy Rule from eCFR / Cornell Law."""
    corpus_id = "hipaa"
    # eCFR has the full regulation text as static HTML
    sources = [
        ("https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E", "hipaa_ecfr_privacy.html"),
        ("https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C", "hipaa_ecfr_security.html"),
        ("https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-160", "hipaa_ecfr_general.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  HIPAA source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "legal_regulatory",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: Cornell Law
    logger.info("HIPAA eCFR failed. Trying Cornell Law...")
    fallback_sources = [
        ("https://www.law.cornell.edu/cfr/text/45/part-164/subpart-E", "hipaa_cornell_privacy.html"),
        ("https://www.law.cornell.edu/cfr/text/45/part-164/subpart-C", "hipaa_cornell_security.html"),
    ]

    combined_text = []
    for url, filename in fallback_sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            if count_tokens(text) > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "legal_regulatory",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_msft_10k():
    """Download Microsoft 10-K from SEC EDGAR."""
    corpus_id = "msft_10k"
    raw_path = os.path.join(RAW_DIR, "msft_10k.html")

    # Correct accession numbers for Microsoft 10-K filings (CIK 789019)
    urls = [
        # FY2024 10-K
        "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
        # FY2023 10-K
        "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
    ]

    for url in urls:
        if download_url(url, raw_path, timeout=120):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            if count_tokens(text) >= 5000:
                save_processed_document(corpus_id, text, {
                    "corpus_id": corpus_id,
                    "domain": "corporate_finance",
                    "source_url": url,
                    "download_date": str(datetime.date.today()),
                })
                return True
        logger.info("MSFT 10-K URL failed, trying next...")

    return False


def download_goog_10k():
    """Download Alphabet (Google) 10-K from SEC EDGAR."""
    corpus_id = "goog_10k"
    # Alphabet FY2023 10-K filing
    url = "https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm"
    raw_path = os.path.join(RAW_DIR, "goog_10k.html")

    if download_url(url, raw_path, timeout=120):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "corporate_finance",
                "source_url": url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: Alternative Alphabet filing
    logger.info("GOOG 10-K primary failed. Trying alternative...")
    fallback_url = "https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm"
    if download_url(fallback_url, raw_path, timeout=120):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "corporate_finance",
                "source_url": fallback_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_owasp_top10():
    """Download OWASP Top 10 (2021) from GitHub raw Markdown sources."""
    corpus_id = "owasp_top10"
    # owasp.org is JS-rendered; use raw GitHub markdown files instead
    base = "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/en"
    pages = [
        f"{base}/A00_2021_Introduction.md",
        f"{base}/A01_2021-Broken_Access_Control.md",
        f"{base}/A02_2021-Cryptographic_Failures.md",
        f"{base}/A03_2021-Injection.md",
        f"{base}/A04_2021-Insecure_Design.md",
        f"{base}/A05_2021-Security_Misconfiguration.md",
        f"{base}/A06_2021-Vulnerable_and_Outdated_Components.md",
        f"{base}/A07_2021-Identification_and_Authentication_Failures.md",
        f"{base}/A08_2021-Software_and_Data_Integrity_Failures.md",
        f"{base}/A09_2021-Security_Logging_and_Monitoring_Failures.md",
        f"{base}/A10_2021-Server-Side_Request_Forgery_(SSRF).md",
    ]

    combined_text = []
    for url in pages:
        filename = url.split("/")[-1]
        raw_path = os.path.join(RAW_DIR, f"owasp_{filename}")
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                text = f.read()  # Markdown is already plain text
            tokens = count_tokens(text)
            logger.info(f"  OWASP source {filename}: {tokens} tokens")
            if tokens > 100:
                combined_text.append(text)
            time.sleep(0.3)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "technical_standard",
                "source_url": "https://github.com/OWASP/Top10",
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_nist_csf():
    """Download NIST Cybersecurity Framework."""
    corpus_id = "nist_csf"
    # Try HTML version first
    url = "https://csrc.nist.gov/pubs/cswp/29/the-nist-cybersecurity-framework-20/final"
    raw_path = os.path.join(RAW_DIR, "nist_csf.html")

    if download_url(url, raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "technical_standard",
                "source_url": url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Try PDF version
    pdf_url = "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"
    pdf_path = os.path.join(RAW_DIR, "nist_csf.pdf")
    if download_url(pdf_url, pdf_path, timeout=120):
        text = extract_text_from_pdf(pdf_path)
        if count_tokens(text) >= 5000:
            save_processed_document(corpus_id, text, {
                "corpus_id": corpus_id,
                "domain": "technical_standard",
                "source_url": pdf_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_who_covid():
    """Download WHO COVID-19 clinical guidelines from multiple sources."""
    corpus_id = "who_covid"
    sources = [
        ("https://www.who.int/publications/i/item/WHO-2019-nCoV-therapeutics-2024.3", "who_covid_therapeutics.html"),
        ("https://www.who.int/publications/i/item/WHO-2019-nCoV-clinical-2024.1", "who_covid_clinical.html"),
        ("https://www.cdc.gov/covid/hcp/clinical-care/index.html", "cdc_covid_clinical.html"),
        ("https://www.cdc.gov/covid/hcp/clinical-care/clinical-considerations.html", "cdc_covid_considerations.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK570236/", "ncbi_covid_statpearls.html"),
        ("https://www.ncbi.nlm.nih.gov/books/NBK554776/", "ncbi_covid_features.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  COVID source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "medical_guideline",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_cdc_immunization():
    """Download CDC immunization guidelines."""
    corpus_id = "cdc_immunization"
    sources = [
        # StatPearls: Understanding and Application of CDC Immunization Guidelines
        ("https://www.ncbi.nlm.nih.gov/books/NBK567723/", "ncbi_cdc_imz_guidelines.html"),
        # StatPearls: Immunization (general overview)
        ("https://www.ncbi.nlm.nih.gov/books/NBK459331/", "ncbi_immunization.html"),
        # StatPearls: Safe and Effective Administration of Vaccines
        ("https://www.ncbi.nlm.nih.gov/books/NBK567772/", "ncbi_vaccine_admin.html"),
        # StatPearls: DTaP Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK545173/", "ncbi_dtap.html"),
        # StatPearls: MMR Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK554450/", "ncbi_mmr.html"),
        # StatPearls: Influenza Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK537197/", "ncbi_influenza.html"),
        # StatPearls: Pneumococcal Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK507794/", "ncbi_pneumococcal.html"),
        # StatPearls: HPV Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK562186/", "ncbi_hpv.html"),
        # StatPearls: Hepatitis A Vaccine
        ("https://www.ncbi.nlm.nih.gov/books/NBK554604/", "ncbi_hepa.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  Immunization source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "medical_guideline",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_faa_ac():
    """Download FAA Advisory Circulars / airworthiness standards."""
    corpus_id = "faa_ac"

    # faa.gov blocks scrapers (403). Use regulations.gov PDF + eCFR HTML instead.
    pdf_sources = [
        ("https://downloads.regulations.gov/FAA-2022-1544-0004/attachment_5.pdf", "faa_ac_25_1309.pdf"),
    ]

    # 14 CFR Part 25 airworthiness standards from eCFR (static HTML)
    html_sources = [
        ("https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-D", "faa_ecfr_25d.html"),
        ("https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-E", "faa_ecfr_25e.html"),
        ("https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-F", "faa_ecfr_25f.html"),
    ]

    combined_text = []
    source_url = ""

    for url, filename in pdf_sources:
        pdf_path = os.path.join(RAW_DIR, filename)
        if download_url(url, pdf_path, timeout=180):
            text = extract_text_from_pdf(pdf_path)
            tokens = count_tokens(text)
            logger.info(f"  FAA source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(1)

    for url, filename in html_sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  FAA source {filename}: {tokens} tokens")
            if tokens > 200:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(0.5)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "engineering_spec",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


def download_nasa_std_2():
    """Download NASA Systems Engineering Handbook (additional NASA standard)."""
    corpus_id = "nasa_std_2"

    # Primary: NTRS direct PDF download (NASA/SP-2016-6105 Rev 2)
    pdf_sources = [
        ("https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf", "nasa_se_handbook.pdf"),
    ]

    # Fallback HTML: NODIS standards
    html_sources = [
        ("https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7123_001C_&page_name=all", "nasa_npr_7123_1c.html"),
        ("https://nodis3.gsfc.nasa.gov/displayAll.cfm?Internal_ID=N_PR_7120_005F_&page_name=all", "nasa_npr_7120_5f.html"),
    ]

    combined_text = []
    source_url = ""

    # Try PDF first (large, reliable source)
    for url, filename in pdf_sources:
        pdf_path = os.path.join(RAW_DIR, filename)
        if download_url(url, pdf_path, timeout=180):
            text = extract_text_from_pdf(pdf_path)
            tokens = count_tokens(text)
            logger.info(f"  NASA-2 source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(1)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "engineering_spec",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    # Fallback: HTML sources
    logger.info("NASA-2 PDF failed. Trying NODIS HTML sources...")
    for url, filename in html_sources:
        raw_path = os.path.join(RAW_DIR, filename)
        if download_url(url, raw_path, timeout=120):
            with open(raw_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            tokens = count_tokens(text)
            logger.info(f"  NASA-2 source {filename}: {tokens} tokens")
            if tokens > 500:
                combined_text.append(text)
                if not source_url:
                    source_url = url
        time.sleep(1)

    if combined_text:
        full_text = "\n\n".join(combined_text)
        if count_tokens(full_text) >= 5000:
            save_processed_document(corpus_id, full_text, {
                "corpus_id": corpus_id,
                "domain": "engineering_spec",
                "source_url": source_url,
                "download_date": str(datetime.date.today()),
            })
            return True

    return False


# --- Master download function ---

DOWNLOAD_FUNCTIONS = {
    # Original 8
    "aws_s3": download_aws_s3,
    "postgresql": download_postgresql,
    "arduino": download_arduino,
    "gdpr": download_gdpr,
    "nist_800_53": download_nist,
    "basel_iii": download_basel,
    "who_malaria": download_who,
    "nasa_std": download_nasa,
    # New 16 (n=3 expansion)
    "azure_blob": download_azure_blob,
    "gcs": download_gcs,
    "mysql": download_mysql,
    "sqlite": download_sqlite,
    "raspberry_pi": download_raspberry_pi,
    "esp32": download_esp32,
    "ccpa": download_ccpa,
    "hipaa": download_hipaa,
    "msft_10k": download_msft_10k,
    "goog_10k": download_goog_10k,
    "owasp_top10": download_owasp_top10,
    "nist_csf": download_nist_csf,
    "who_covid": download_who_covid,
    "cdc_immunization": download_cdc_immunization,
    "faa_ac": download_faa_ac,
    "nasa_std_2": download_nasa_std_2,
}


def download_all():
    """Download all document corpora. Skips corpora that already have processed files."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results = {}
    for corpus_id, download_fn in DOWNLOAD_FUNCTIONS.items():
        # Skip if already downloaded and has sufficient tokens
        text_path = os.path.join(PROCESSED_DIR, f"{corpus_id}.txt")
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                text = f.read()
            tokens = count_tokens(text)
            if tokens >= 5000:
                logger.info(f"Skipping {corpus_id}: already exists ({tokens} tokens)")
                results[corpus_id] = True
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {corpus_id} ({CORPORA[corpus_id]['description']})")
        logger.info(f"{'='*60}")
        try:
            success = download_fn()
            results[corpus_id] = success
            if success:
                # Verify token count
                if os.path.exists(text_path):
                    with open(text_path, "r") as f:
                        text = f.read()
                    tokens = count_tokens(text)
                    logger.info(f"  -> {corpus_id}: {tokens} tokens")
                    if tokens < 15000:
                        logger.warning(f"  -> WARNING: {corpus_id} has only {tokens} tokens (min 15000)")
        except Exception as e:
            logger.error(f"  -> FAILED: {corpus_id}: {e}")
            results[corpus_id] = False

    # Summary
    successes = sum(1 for v in results.values() if v)
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary: {successes}/{len(results)} corpora successful")
    for cid, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info(f"  {cid}: {status}")
    logger.info(f"{'='*60}")

    if successes < 6:
        logger.error(f"Only {successes} corpora available. Minimum 6 required.")

    return results


if __name__ == "__main__":
    download_all()
