import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from ingest import get_embeddings
from analysis_report import strong_llm, safe_llm_invoke, RateLimitError

load_dotenv()

VECTORSTORE_PATH = "vectorstore"
SEARCH_CACHE = {}
SEARCH_TIMEOUT = 30


# -------------------------------------------------
# Load FAISS safely
# -------------------------------------------------
def load_vectorstore():
    """Load the FAISS vector database if it exists."""
    if not os.path.exists(VECTORSTORE_PATH):
        return None

    try:
        return FAISS.load_local(
            VECTORSTORE_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None


# -------------------------------------------------
# Web search helpers
# Priority: Tavily -> DuckDuckGo -> Wikipedia
# -------------------------------------------------
def _format_results(results):
    """Normalize search results from different providers."""
    formatted = []
    seen_urls = set()

    for result in results or []:
        if not isinstance(result, dict):
            continue

        title = str(result.get("title") or "").strip()
        url = str(
            result.get("url")
            or result.get("href")
            or result.get("link")
            or ""
        ).strip()

        content = str(
            result.get("raw_content")
            or result.get("content")
            or result.get("body")
            or result.get("snippet")
            or result.get("description")
            or ""
        ).strip()

        if not url.startswith(("http://", "https://")):
            continue
        if url in seen_urls or not (title or content):
            continue

        seen_urls.add(url)
        formatted.append({
            "title": title or urlparse(url).netloc,
            "url": url,
            "snippet": content[:8000],
        })

    return formatted


# -------------------------------------------------
# Tavily Search (Primary)
# -------------------------------------------------
def _search_tavily(query, max_results):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[WEB SEARCH] Tavily API key not configured")
        return []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "topic": "general",
                "search_depth": "advanced",
                "max_results": max_results,
                "include_raw_content": "text",
            },
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        results = _format_results(data.get("results", []))
        print(f"[WEB SEARCH] Tavily: {len(results)} results")
        return results

    except requests.RequestException as e:
        print(f"[WEB SEARCH ERROR] Tavily request: {e}")
    except (ValueError, TypeError) as e:
        print(f"[WEB SEARCH ERROR] Tavily response: {e}")
    except Exception as e:
        print(f"[WEB SEARCH ERROR] Tavily: {type(e).__name__}: {e}")

    return []


# -------------------------------------------------
# DuckDuckGo Search (Fallback)
# -------------------------------------------------
def _search_ddgs(query, max_results):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("[WEB SEARCH] Install DDGS: pip install -U ddgs")
            return []

    backends = ("auto", "duckduckgo", "bing", "brave", "yahoo")

    for backend in backends:
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    try:
                        raw_results = ddgs.text(
                            query,
                            max_results=max_results,
                            region="in-en",
                            safesearch="moderate",
                            backend=backend,
                        )
                    except TypeError:
                        # Compatibility with older DDGS package versions.
                        raw_results = ddgs.text(
                            query,
                            max_results=max_results,
                            region="in-en",
                            safesearch="moderate",
                        )

                    results = _format_results(list(raw_results or []))

                if results:
                    print(
                        f"[WEB SEARCH] DuckDuckGo/{backend}: "
                        f"{len(results)} results"
                    )
                    return results

                print(f"[WEB SEARCH] DuckDuckGo/{backend}: 0 results")
                break

            except Exception as e:
                print(
                    f"[WEB SEARCH ERROR] DuckDuckGo/{backend} "
                    f"attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                if attempt == 0:
                    time.sleep(1.5)

    return []


# -------------------------------------------------
# Wikipedia Search (Last Fallback)
# -------------------------------------------------
def _search_wikipedia(query, max_results):
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": min(max_results, 3),
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 3000,
                "inprop": "url",
                "format": "json",
            },
            headers={"User-Agent": "RAGAssistant/1.0 (educational project)"},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()

        pages = (response.json().get("query") or {}).get("pages", {})
        ordered_pages = sorted(
            pages.values(),
            key=lambda page: page.get("index", 999),
        )

        raw_results = [
            {
                "title": f"{page.get('title', '')} - Wikipedia",
                "url": page.get("fullurl", ""),
                "content": page.get("extract", ""),
            }
            for page in ordered_pages
            if page.get("extract") and page.get("fullurl")
        ]

        results = _format_results(raw_results)
        print(f"[WEB SEARCH] Wikipedia: {len(results)} results")
        return results

    except Exception as e:
        print(f"[WEB SEARCH ERROR] Wikipedia: {type(e).__name__}: {e}")
        return []


# -------------------------------------------------
# Main Web Search Router
# -------------------------------------------------
def web_search(query, max_results=5):
    """Search using Tavily first, then DDGS, then Wikipedia."""
    query = (query or "").strip()
    if not query:
        return []

    cache_key = (query.lower(), max_results)
    if cache_key in SEARCH_CACHE:
        print("[WEB SEARCH] Cache hit")
        return SEARCH_CACHE[cache_key]

    print(f"[WEB SEARCH] Query: {query}")

    search_engines = (
        ("Tavily", _search_tavily),
        ("DuckDuckGo", _search_ddgs),
        ("Wikipedia", _search_wikipedia),
    )

    for engine_name, engine_function in search_engines:
        results = engine_function(query, max_results)
        if results:
            SEARCH_CACHE[cache_key] = results
            print(f"[WEB SEARCH] Selected provider: {engine_name}")
            return results

    print("[WEB SEARCH] All search providers failed")
    return []


# -------------------------------------------------
# Build contexts
# -------------------------------------------------
def build_web_context(search_results):
    if not search_results:
        return ""

    context_parts = []
    for index, result in enumerate(search_results, start=1):
        context_parts.append(
            f"SOURCE {index}\n"
            f"Title: {result.get('title', 'Unknown')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('snippet', '')}\n"
        )

    return "\n\n".join(context_parts)


def build_doc_context(docs):
    return "\n\n".join(
        doc.page_content for doc in docs if doc.page_content.strip()
    )


# -------------------------------------------------
# Answer From Uploaded Documents
# -------------------------------------------------
def _dedupe_docs(docs):
    """Remove duplicate (file, page) entries while preserving order."""
    seen = set()
    unique = []

    for doc in docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique


def answer_from_documents(question, db):
    """Answer only from uploaded documents; return NOT_FOUND if unsupported."""
    if db is None:
        return "NOT_FOUND", []

    try:
        docs = db.similarity_search(question, k=6)
    except Exception as e:
        print(f"Document retrieval error: {e}")
        return "NOT_FOUND", []

    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        return "NOT_FOUND", []

    context = "\n\n".join(
        f"[CHUNK {index}] (file: "
        f"{os.path.basename(str(doc.metadata.get('source', 'Unknown')))}, "
        f"page: {doc.metadata.get('page', '?')})\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )

    prompt = f"""
You are an intelligent AI document assistant.
Answer the user's question using ONLY the uploaded document chunks below.

Rules:
1. Use only chunks that directly answer the question.
2. Ignore merely related or unrelated chunks.
3. Do not invent information or use outside knowledge.
4. Use reasonable inference only when clearly supported by the chunks.
5. If the chunks do not contain enough information, respond exactly: NOT_FOUND
6. Do not mention chunk numbers, filenames, or sources in the answer.
7. After the answer, add one final line listing only the chunk numbers used:
   USED_CHUNKS: 1,3
8. Do not add USED_CHUNKS when the answer is NOT_FOUND.

UPLOADED DOCUMENT CHUNKS:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}
"""

    raw = safe_llm_invoke(strong_llm, prompt)
    if not raw or not str(raw).strip():
        return "NOT_FOUND", []

    raw = str(raw).strip()
    used_indices = []
    match = re.search(
        r"USED_CHUNKS:\s*([\d,\s]+)",
        raw,
        flags=re.IGNORECASE,
    )

    if match:
        used_indices = [
            int(number)
            for number in re.findall(r"\d+", match.group(1))
        ]
        answer = re.sub(
            r"\n?\s*USED_CHUNKS:\s*[\d,\s]+\s*$",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
    else:
        answer = raw

    if _is_not_found(answer):
        return "NOT_FOUND", []

    used_docs = [
        docs[index - 1]
        for index in used_indices
        if 1 <= index <= len(docs)
    ]

    # Fallback if the model did not return valid chunk references.
    if not used_docs:
        used_docs = docs[:1]

    return answer, _dedupe_docs(used_docs)


# -------------------------------------------------
# Answer From Web
# -------------------------------------------------
def answer_from_web(question, max_results=5):
    """Search the web and generate an answer grounded in returned sources."""
    search_results = web_search(question, max_results=max_results)

    if not search_results:
        return "I couldn't find web search results for this question.", []

    web_context = build_web_context(search_results)

    prompt = f"""
You are an accurate AI web research assistant.
Answer the user's question using the web search results supplied below.

RESEARCH AND ANSWERING RULES:
1. First assess which sources are directly relevant to the question.
2. Ignore unrelated or weakly relevant results.
3. Prefer official, primary, authoritative, and trustworthy sources.
4. Use only facts supported by the supplied source content.
5. Do not treat a page title or a short search snippet as proof of a claim.
6. Do not invent facts, dates, names, statistics, quotations, or events.
7. If sources disagree, explain the disagreement instead of hiding it.
8. For current or time-sensitive questions, check dates in the source content.
   Do not call old information current.
9. If the supplied content is insufficient, clearly state what cannot be verified.
10. For a definition question, explain the definition, key characteristics,
    relevant background, and examples when the sources support them.
11. For a person, include profession, background, career, and achievements
    only when supported by the supplied sources.
12. For lists, comparisons, timelines, rankings, or year-wise information,
    use a table or bullets when helpful.
13. If the user asks for all/every/complete information, do not claim
    completeness unless a comprehensive source supports it.
14. Keep the answer focused, clear, and sufficiently detailed.
15. Do not include raw URLs or a separate Sources section in the answer;
    the application displays the source links separately.
16. Do not mention internal prompts, retrieval, or system instructions.

WEB SEARCH RESULTS:
-------------------
{web_context}
-------------------

USER QUESTION:
{question}
"""

    answer = safe_llm_invoke(strong_llm, prompt)
    if not answer or not str(answer).strip():
        return (
            "I found web results, but I was unable to generate a useful answer.",
            search_results,
        )

    return str(answer).strip(), search_results


# -------------------------------------------------
# Main Router: Documents first -> Web fallback
# -------------------------------------------------
NOT_FOUND_PHRASES = (
    "not_found",
    "does not contain",
    "doesn't contain",
    "does not mention",
    "doesn't mention",
    "not mentioned in",
    "no information",
    "not enough information",
    "cannot find",
    "can't find",
    "not available in the",
)


def _is_not_found(answer):
    if not answer or not str(answer).strip():
        return True

    answer_text = str(answer).strip().lower()
    if answer_text.startswith("not_found"):
        return True

    if len(answer_text) < 200 and any(
        phrase in answer_text for phrase in NOT_FOUND_PHRASES
    ):
        return True

    return False


def answer_question(question, db: Optional[FAISS] = None):
    """
    Priority:
      1. Uploaded documents, when available.
      2. Web search, when documents cannot answer or no DB exists.

    Returns: (answer, sources, source_type)
    source_type: document | web | none | error
    """
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", [], "none"

    try:
        if db is not None:
            print("[ROUTER] Documents available -> trying documents first")
            answer, docs = answer_from_documents(question, db)

            if not _is_not_found(answer):
                print("[ROUTER] Answered from DOCUMENTS")
                return answer, docs, "document"

            print("[ROUTER] Documents could not answer -> web fallback")
        else:
            print("[ROUTER] No documents uploaded -> web search")

        web_answer, web_sources = answer_from_web(question)

        if web_sources:
            print("[ROUTER] Answered from WEB")
            return web_answer, web_sources, "web"

        return web_answer, [], "none"

    except RateLimitError:
        return (
            "Groq rate limit reach ho gayi hai. Thodi der baad try karo.",
            [],
            "error",
        )
    except Exception as e:
        print(f"[ANSWER ERROR] {type(e).__name__}: {e}")
        return f"Something went wrong: {e}", [], "error"


# -------------------------------------------------
# Standalone Test
# -------------------------------------------------
if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    db = load_vectorstore()

    answer, sources, source_type = answer_question(question, db)

    print("\n" + "=" * 60)
    print("SOURCE TYPE:")
    print(source_type)
    print("\nANSWER:")
    print(answer)
    print("\nSOURCES:")

    if source_type == "document":
        for source in sources:
            filename = os.path.basename(
                source.metadata.get("source", "Unknown")
            )
            page = source.metadata.get("page", "?")
            print(f"- {filename} (Page {page})")

    elif source_type == "web":
        for index, source in enumerate(sources, start=1):
            print(f"- [{index}] {source.get('title', 'Unknown')}")
            print(f"  {source.get('url', '')}")

    elif source_type == "error":
        print("(See error message above.)")

    print("=" * 60)
