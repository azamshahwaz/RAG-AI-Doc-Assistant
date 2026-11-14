import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ============================================================
# Environment and Groq configuration
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "").strip()

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your .env file before starting the app."
    )

fast_llm = ChatGroq(
    model=PRIMARY_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1200,
)

strong_llm = ChatGroq(
    model=PRIMARY_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1500,
)

# Fallback is optional. Configure it only with a model currently enabled
# for your Groq account. It uses the same API key and therefore may share
# account-level limits.
fallback_fast_llm = None
fallback_strong_llm = None

if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
    fallback_fast_llm = ChatGroq(
        model=FALLBACK_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1200,
    )
    fallback_strong_llm = ChatGroq(
        model=FALLBACK_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1500,
    )


class RateLimitError(Exception):
    """Raised when Groq rejects a request because a usage limit was reached."""

    def __init__(self, message, retry_after=None, limit_kind="unknown"):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_kind = limit_kind


def _extract_retry_after(error_text: str):
    """Extract a retry delay, in seconds, from common Groq error messages."""
    patterns = (
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retry[- ]after[:\s]+([0-9]+(?:\.[0-9]+)?)",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?",
    )
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.0, float(match.group(1)))
            except ValueError:
                pass

    # Some Groq messages express the delay as "1m2.5s".
    match = re.search(
        r"try again in\s+(?:(\d+)m)?\s*([0-9]+(?:\.[0-9]+)?)s",
        error_text,
        flags=re.IGNORECASE,
    )
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return minutes * 60 + seconds

    return None


def _classify_rate_limit(error_text: str):
    text = error_text.lower()

    if any(term in text for term in (
        "tokens per day", "token limit for the day", "daily token",
        "requests per day", "per day limit", "daily limit",
    )):
        return "daily"

    if any(term in text for term in (
        "tokens per minute", "tpm", "requests per minute", "rpm",
        "per minute",
    )):
        return "minute"

    return "unknown"


def _is_rate_limit_error(error: Exception):
    text = str(error).lower()
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    return (
        status_code == 429
        or "429" in text
        or "rate_limit" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def _fallback_for(llm):
    """Choose the matching optional fallback model for fast/strong calls."""
    if llm is fast_llm:
        return fallback_fast_llm
    if llm is strong_llm:
        return fallback_strong_llm
    return None


def safe_llm_invoke(
    llm,
    prompt,
    retries=2,
    base_delay=2,
    max_rate_limit_wait=12,
):
    """
    Invoke a LangChain chat model safely.

    - Retries temporary network/server errors with exponential backoff.
    - For short RPM/TPM cooldowns, waits for the provider's retry hint.
    - Does not repeatedly retry daily quota exhaustion.
    - Optionally tries GROQ_FALLBACK_MODEL once after a primary 429.
    """
    last_error = None

    for attempt in range(max(0, retries) + 1):
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

        except Exception as exc:
            last_error = exc
            error_text = str(exc)

            if _is_rate_limit_error(exc):
                retry_after = _extract_retry_after(error_text)
                limit_kind = _classify_rate_limit(error_text)

                # If a fallback model is configured, try it once. It may
                # still be subject to the same account-level quota.
                fallback_llm = _fallback_for(llm)
                if fallback_llm is not None:
                    try:
                        print(
                            "[GROQ] Primary model rate-limited. "
                            f"Trying fallback model: {FALLBACK_MODEL}"
                        )
                        fallback_response = fallback_llm.invoke(prompt)
                        fallback_content = getattr(
                            fallback_response, "content", fallback_response
                        )
                        return (
                            fallback_content.strip()
                            if isinstance(fallback_content, str)
                            else str(fallback_content).strip()
                        )
                    except Exception as fallback_exc:
                        print(
                            "[GROQ] Fallback model also failed: "
                            f"{type(fallback_exc).__name__}: {fallback_exc}"
                        )
                        error_text = (
                            f"Primary model error: {error_text}\n"
                            f"Fallback model error: {fallback_exc}"
                        )
                        retry_after = (
                            _extract_retry_after(str(fallback_exc))
                            or retry_after
                        )
                        limit_kind = _classify_rate_limit(error_text)

                # Daily limits should not be retried in a tight loop.
                if limit_kind == "daily":
                    raise RateLimitError(
                        "Groq daily usage quota has been exhausted. "
                        "Wait until the quota resets, or use an API/model "
                        "with available quota.",
                        retry_after=retry_after,
                        limit_kind=limit_kind,
                    ) from exc

                # Retry only when the provider's cooldown is short enough.
                if (
                    retry_after is not None
                    and retry_after <= max_rate_limit_wait
                    and attempt < retries
                ):
                    wait_seconds = max(1.0, retry_after + 1.0)
                    print(
                        f"[GROQ] Temporary rate limit. Retrying in "
                        f"{wait_seconds:.1f} seconds "
                        f"(attempt {attempt + 1}/{retries})."
                    )
                    time.sleep(wait_seconds)
                    continue

                retry_note = (
                    f" Provider retry hint: {retry_after:g} seconds."
                    if retry_after is not None
                    else ""
                )
                raise RateLimitError(
                    "Groq rate limit reached for the configured model."
                    f"{retry_note} If this is an RPM/TPM limit, wait for "
                    "the cooldown. If it is a daily quota, wait for reset "
                    "or configure a provider/model with available quota.",
                    retry_after=retry_after,
                    limit_kind=limit_kind,
                ) from exc

            # Retry transient network/server failures with exponential backoff.
            if attempt < retries:
                wait_seconds = base_delay * (2 ** attempt)
                print(
                    f"[LLM] Temporary error: {type(exc).__name__}. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

    raise last_error


# ============================================================
# Supported Document Types
# ============================================================

VALID_DOCUMENT_TYPES = [
    "Research Paper",
    "Resume / CV",
    "Project Report",
    "Technical Documentation",
    "Invoice / Financial Document",
    "Dataset / CSV / Excel",
    "Presentation / PPT",
    "Academic Notes / Textbook",
    "Legal Document",
    "Business Document",
    "Certificate",
    "User Manual",
    "General Document",
]


# ============================================================
# Chunk Settings
# ============================================================

CHUNK_SIZE = 15000
CHUNK_OVERLAP = 400
CHUNK_DELAY_SECONDS = float(os.getenv("CHUNK_DELAY_SECONDS", "1.0"))


# ============================================================
# Split Text Into Chunks
# ============================================================

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []

    text = text.strip()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length:
            last_newline = chunk.rfind("\n")
            if last_newline > chunk_size * 0.7:
                end = start + last_newline
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


# ============================================================
# Detect Document Type
# ============================================================

def detect_document_type(text):
    prompt = f"""
You are an intelligent document classification system.

Identify the type of the provided document.

Choose EXACTLY ONE category:

1. Research Paper
2. Resume / CV
3. Project Report
4. Technical Documentation
5. Invoice / Financial Document
6. Dataset / CSV / Excel
7. Presentation / PPT
8. Academic Notes / Textbook
9. Legal Document
10. Business Document
11. Certificate
12. User Manual
13. General Document

Classification guidelines:

Research Paper: Abstract, Introduction, Methodology, Experiments, Results, Discussion, References
Resume / CV: Name, Contact information, Education, Skills, Experience, Projects, Certifications
Project Report: Objectives, Requirements, Architecture, Implementation, Technologies, Modules, Testing, Results, Future scope
Technical Documentation: APIs, Technical specifications, Configuration, Architecture, Developer documentation
Invoice / Financial Document: Invoice, Billing, Items, Quantity, Price, Tax, Total amount, Financial transactions
Dataset / CSV / Excel: Rows, Columns, Records, Variables, Tabular data
Presentation / PPT: Slides, Slide titles, Bullet points, Presentation topics
Academic Notes / Textbook: Educational explanations, Definitions, Concepts, Chapters, Examples, Study material
Legal Document: Contracts, Agreements, Clauses, Terms, Legal provisions
Business Document: Business plans, Proposals, Company reports, Business strategies
Certificate: Certificate title, Recipient, Issuing organization, Date, Certification statement
User Manual: Installation, Setup, Usage instructions, Troubleshooting, Product instructions
General Document: Use only when no other category clearly fits.

Return ONLY the category name.

DOCUMENT CONTENT:
{text[:6000]}
"""

    try:
        document_type = safe_llm_invoke(fast_llm, prompt)
        for valid_type in VALID_DOCUMENT_TYPES:
            if valid_type.lower() in document_type.lower():
                return valid_type
        return "General Document"
    except RateLimitError:
        raise
    except Exception:
        return "General Document"


# ============================================================
# Analyze Individual Chunk
# ============================================================

def analyze_chunk(chunk, chunk_number, total_chunks, document_type):
    prompt = f"""
You are analyzing part {chunk_number} of {total_chunks} of a document.

Document Type:
{document_type}

Analyze ONLY the provided section.
Do NOT invent information.
Extract the most important information from this section.

Focus on:
- Important concepts, definitions, topics, names, dates, numbers
- Formulas, technologies, organizations, results, examples
- Important statements, conclusions, technical details

If something is not present, do not invent it.

Create a concise structured summary that can later be combined
with summaries from other sections.

DOCUMENT SECTION:
{chunk}
"""

    try:
        return safe_llm_invoke(fast_llm, prompt)
    except RateLimitError:
        raise
    except Exception as exc:
        return f"Unable to analyze section {chunk_number}: {str(exc)}"


# ============================================================
# Generate Final Report
# ============================================================

def generate_final_report(summaries, filename, document_type):
    combined_summaries = "\n\n".join(
        f"===== SECTION {i + 1} SUMMARY =====\n{summary}"
        for i, summary in enumerate(summaries)
    )

    prompt = f"""
You are an advanced AI document analysis assistant.

Generate a final analysis report for ONE document.

==================================================
DOCUMENT INFORMATION
==================================================

Filename:
{filename}

Document Type:
{document_type}

==================================================
IMPORTANT RULES
==================================================

Use ONLY the information contained in the section summaries below.
Do NOT invent facts.
Do NOT assume missing information.
Do NOT change numerical values.
Do NOT change names.

If information is unavailable, write:
"Not specified in the document."

The report must be based on the actual document type.
Create approximately 6-10 useful sections.

Use Markdown with headings, subheadings, bullet points,
numbered lists, tables when useful, important numbers,
names, dates, technologies, organizations, results and
technical details.

==================================================
DOCUMENT-TYPE ANALYSIS
==================================================

For Academic Notes / Textbook, focus on: Subject/topic, Main concepts,
Definitions, Important theories, Examples, Key points, Important formulas,
Learning summary.

For Research Paper, focus on: Research objective, Problem statement,
Methodology, Dataset/materials, Experiments, Results, Findings,
Limitations, Conclusion.

For Resume / CV, focus on: Candidate overview, Education, Skills,
Experience, Projects, Certifications, Achievements, Technologies.

For Project Report, focus on: Project overview, Problem statement,
Objectives, Requirements, Technologies, Architecture, Modules,
Implementation, Testing, Results, Future scope.

For Technical Documentation, focus on: Purpose, Architecture,
Technical specifications, APIs, Configuration, Dependencies,
Procedures, Troubleshooting.

For Legal Document, focus on: Purpose, Parties, Clauses, Obligations,
Rights, Terms, Conditions, Important provisions.

For Business Document, focus on: Business purpose, Objectives, Strategy,
Products/services, Financial information, Risks, Opportunities, Conclusions.

For User Manual, focus on: Product/system overview, Requirements,
Installation, Configuration, Usage, Procedures, Troubleshooting,
Safety, Maintenance.

For Certificate, focus on: Certificate type, Recipient, Issuing
organization, Achievement, Date, Validity information.

For Invoice / Financial Document, focus on: Parties, Invoice details,
Items/services, Quantity, Prices, Taxes, Discounts, Total, Payment information.

For Dataset / CSV / Excel, focus on: Dataset overview, Records, Columns,
Variables, Data types, Categories, Numerical information, Patterns,
Potential use cases.

For Presentation / PPT, focus on: Topic, Objective, Main themes,
Important concepts, Key messages, Data/statistics, Conclusions.

==================================================
OUTPUT FORMAT
==================================================

Start EXACTLY with:

# Document Analysis Report

## Document Information

| Field | Details |
|---|---|
| Filename | {filename} |
| Document Type | {document_type} |

Then write:

## Overview

Provide a clear overview of the document.

Then create the most relevant sections based on the document type.

Finally end with:

## Overall Summary

Explain what the document is about, its main purpose, most important
information, major findings, and overall conclusion.

==================================================
SECTION SUMMARIES
==================================================

{combined_summaries}

==================================================
END
==================================================
"""

    try:
        report = safe_llm_invoke(strong_llm, prompt)
        if not report:
            return (
                f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
                "The AI was unable to generate an analysis report."
            )
        return report

    except RateLimitError as exc:
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"## Document Type\n\n{document_type}\n\n"
            f"### Rate limit reached\n\n{str(exc)}"
        )

    except Exception as exc:
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"## Document Type\n\n{document_type}\n\n"
            "Unable to generate the final analysis report.\n\n"
            f"### Error\n\n{str(exc)}"
        )


# ============================================================
# Generate Dynamic Document Analysis Report
# ============================================================

def generate_analysis_report(text, filename="Unknown Document"):
    """
    Generate a detailed AI analysis report for one document.

    PDF/text -> chunking -> individual chunk analysis (fast model)
    -> summary aggregation -> final report (strong model).

    A single-chunk document skips the intermediate summary call.
    """
    if not text or not text.strip():
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            "No readable text was found in this document."
        )

    try:
        document_type = detect_document_type(text)
    except RateLimitError as exc:
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"### Rate limit reached\n\n{str(exc)}"
        )

    chunks = split_text_into_chunks(text)
    if not chunks:
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            "No readable text was found in this document."
        )

    total_chunks = len(chunks)

    try:
        if total_chunks == 1:
            # Skip one intermediate LLM request for small documents.
            summaries = [chunks[0][:6000]]
        else:
            summaries = []
            for index, chunk in enumerate(chunks):
                print(
                    f"Analyzing document section "
                    f"{index + 1}/{total_chunks}..."
                )
                summary = analyze_chunk(
                    chunk=chunk,
                    chunk_number=index + 1,
                    total_chunks=total_chunks,
                    document_type=document_type,
                )
                summaries.append(summary)

                # A configurable pause helps reduce RPM bursts.
                if index < total_chunks - 1 and CHUNK_DELAY_SECONDS > 0:
                    time.sleep(CHUNK_DELAY_SECONDS)

    except RateLimitError as exc:
        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"## Document Type\n\n{document_type}\n\n"
            f"### Rate limit reached\n\n{str(exc)}"
        )

    print("Generating final document analysis report...")
    return generate_final_report(
        summaries=summaries,
        filename=filename,
        document_type=document_type,
    )
