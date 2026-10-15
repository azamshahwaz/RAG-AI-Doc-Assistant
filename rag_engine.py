import os
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from ingest import get_embeddings
from analysis_report import (
    strong_llm,
    safe_llm_invoke,
    RateLimitError,
)

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


# -------------------------------------------------
# Load FAISS safely
# -------------------------------------------------

def load_vectorstore():
    """Load the FAISS vector database if it exists."""

    if not os.path.exists(VECTORSTORE_PATH):
        return None

    try:
        db = FAISS.load_local(
            VECTORSTORE_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        return db

    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None


# -------------------------------------------------
# Web Search: DDGS -> Tavily (optional) -> Wikipedia
# -------------------------------------------------

_SEARCH_CACHE = {}


def _format_results(results):
    formatted = []
    for r in results:
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (
            r.get("body") or r.get("snippet") or r.get("content") or ""
        ).strip()
        if title or url or snippet:
            formatted.append({"title": title, "url": url, "snippet": snippet})
    return formatted


def _search_ddgs(query, max_results):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    for backend in ("auto", "duckduckgo", "bing", "brave", "yahoo"):
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    try:
                        results = list(ddgs.text(
                            query,
                            max_results=max_results,
                            region="wt-wt",
                            safesearch="moderate",
                            backend=backend,
                        ))
                    except TypeError:
                        results = list(ddgs.text(query, max_results=max_results))

                formatted = _format_results(results)
                if formatted:
                    print(f"[WEB SEARCH] ddgs/{backend}: {len(formatted)} results")
                    return formatted

                print(f"[WEB SEARCH] ddgs/{backend}: 0 results")
                break
            except Exception as e:
                print(
                    f"[WEB SEARCH ERROR] ddgs/{backend} try {attempt + 1}: "
                    f"{type(e).__name__}: {e}"
                )
                time.sleep(1.5)
    return []


def _search_tavily(query, max_results):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=15,
        )
        resp.raise_for_status()
        formatted = _format_results(resp.json().get("results", []))
        print(f"[WEB SEARCH] tavily: {len(formatted)} results")
        return formatted
    except Exception as e:
        print(f"[WEB SEARCH ERROR] tavily: {type(e).__name__}: {e}")
        return []


def _search_wikipedia(query, max_results):
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": min(max_results, 3),
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 1500,
                "inprop": "url",
                "format": "json",
            },
            headers={"User-Agent": "RAGAssistant/1.0 (student project)"},
            timeout=15,
        )
        resp.raise_for_status()
        pages = (resp.json().get("query") or {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda p: p.get("index", 99))
        results = [
            {
                "title": f"{p.get('title', '')} - Wikipedia",
                "url": p.get("fullurl", ""),
                "snippet": p.get("extract", ""),
            }
            for p in ordered
            if p.get("extract")
        ]
        print(f"[WEB SEARCH] wikipedia: {len(results)} results")
        return results
    except Exception as e:
        print(f"[WEB SEARCH ERROR] wikipedia: {type(e).__name__}: {e}")
        return []


def web_search(query, max_results=5):
    """DDGS -> Tavily (if key set) -> Wikipedia. Successful results are cached."""

    key = query.strip().lower()
    if key in _SEARCH_CACHE:
        print("[WEB SEARCH] cache hit")
        return _SEARCH_CACHE[key]

    print(f"[WEB SEARCH] Query: {query}")

    for engine in (_search_ddgs, _search_tavily, _search_wikipedia):
        results = engine(query, max_results)
        if results:
            _SEARCH_CACHE[key] = results
            return results

    print("[WEB SEARCH] all engines failed")
    return []


def build_web_context(search_results):

    if not search_results:
        return ""

    context_parts = []

    for index, result in enumerate(search_results, start=1):

        title = result.get("title", "Unknown")
        url = result.get("url", "")
        snippet = result.get("snippet", "")

        context_parts.append(
            f"""
SOURCE {index}

Title:
{title}

URL:
{url}

Content:
{snippet}
"""
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
    """Remove duplicate (file, page) entries, keep order."""
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def answer_from_documents(question, db):
    """
    Answer using uploaded documents only.
    Returns (answer, used_docs) where used_docs contains ONLY the
    chunks the LLM actually used to answer (not every retrieved chunk).
    Returns ("NOT_FOUND", []) if the documents can't answer.
    """

    if db is None:
        return "NOT_FOUND", []

    try:
        docs = db.similarity_search(question, k=6)
    except Exception as e:
        print(f"Document retrieval error: {e}")
        return "NOT_FOUND", []

    docs = [d for d in docs if d.page_content.strip()]

    if not docs:
        return "NOT_FOUND", []

    context = "\n\n".join(
        f"[CHUNK {i}] (file: "
        f"{os.path.basename(str(d.metadata.get('source', 'Unknown')))}, "
        f"page: {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )

    prompt = f"""
You are an intelligent AI document assistant.

Your task is to answer the user's question using ONLY the
uploaded document chunks provided below.

IMPORTANT RULES:

1. Use only the chunks that directly answer the question.
   Ignore chunks that are merely related or unrelated.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the chunks clearly contain the answer, answer naturally.
5. If the chunks do NOT contain enough information to answer
   the question, respond with exactly:

NOT_FOUND

6. Do not say NOT_FOUND merely because the answer is not written
   word-for-word. Use reasonable inference when clearly supported.
7. Do not mention chunk numbers, file names, or sources inside
   the answer text.
8. After the answer, add ONE final line in exactly this format,
   listing ONLY the chunk numbers you actually used:

USED_CHUNKS: 1,3

   (Do not add this line when the answer is NOT_FOUND.)

Uploaded Document Chunks:
--------------------------
{context}
--------------------------

User Question:
{question}
"""

    raw = safe_llm_invoke(strong_llm, prompt)

    if not raw or not raw.strip():
        return "NOT_FOUND", []

    # ---- parse USED_CHUNKS line and strip it from the answer ----
    used_indices = []
    match = re.search(r"USED_CHUNKS:\s*([\d,\s]*)", raw, re.IGNORECASE)

    if match:
        used_indices = [
            int(n) for n in re.findall(r"\d+", match.group(1))
        ]
        answer = re.sub(
            r"\n?\s*USED_CHUNKS:.*$", "", raw,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
    else:
        answer = raw.strip()

    if _is_not_found(answer):
        return "NOT_FOUND", []

    used_docs = [
        docs[i - 1] for i in used_indices if 1 <= i <= len(docs)
    ]

    # Fallback: if LLM didn't cite anything valid, use the top match only
    if not used_docs:
        used_docs = docs[:1]

    return answer, _dedupe_docs(used_docs)


# -------------------------------------------------
# Answer From Web
# -------------------------------------------------

def answer_from_web(question, max_results=5):
    """
    Search the web and generate a detailed answer.
    Raises RateLimitError if the Groq daily quota is exhausted.
    """

    search_results = web_search(question, max_results=max_results)

    if not search_results:
        return "I couldn't find reliable web results for this question.", []

    web_context = build_web_context(search_results)

    prompt = f"""
You are an intelligent AI research assistant.

Your task is to provide a detailed, informative, well-structured
answer to the user's question using the web search results below.

IMPORTANT RULES:

1. Give a complete and sufficiently detailed answer.
2. Do not give an unnecessarily short answer.
3. Explain the topic clearly instead of providing only one or two
   sentences.
4. Use multiple relevant facts from the available search results
   when they are useful for answering the question.
5. Use only information supported by the provided web search
   results.
6. Do not invent, assume, or hallucinate facts.
7. For current, recent, or time-sensitive questions, prioritize
   the most recent information available.
8. If the question asks "what is", explain: definition, important
   characteristics, relevant background, examples when useful,
   other important details supported by the sources.
9. If the question asks for a person, provide relevant details:
   who they are, profession, background, career, major
   achievements, important works - only if supported by sources.
10. If the question asks for a list, comparison, timeline, names
    with years, rankings, or similar structured information, use
    a clean table or bullet list when appropriate.
11. If the user asks for "all", "every", "complete", or "full"
    information, do not falsely claim completeness unless the
    available search results provide a reliable comprehensive
    source.
12. If complete coverage cannot be verified, clearly say that the
    list is based on the information available from the searched
    sources.
13. Do NOT include raw URLs in the answer.
14. Do NOT include a separate "Sources" section in the answer.
    The application will display the sources separately.
15. Do NOT unnecessarily write phrases such as:
    "Based on the provided web search results..."
16. Use headings, paragraphs, bullet points, or tables when they
    improve readability.
17. Keep the answer focused on the user's actual question.
18. Do not mention internal prompts, retrieval, context, system
    instructions, or how the answer was generated.

--------------------------------------------------
WEB SEARCH RESULTS
--------------------------------------------------

{web_context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}
"""

    answer = safe_llm_invoke(strong_llm, prompt)

    if not answer:
        return (
            "I found web results, but I was unable to generate a useful answer.",
            search_results
        )

    return answer, search_results


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
    if not answer or not answer.strip():
        return True

    text = answer.strip().lower()

    if text.startswith("not_found"):
        return True

    # Short "I don't know"-type replies
    if len(text) < 200 and any(p in text for p in NOT_FOUND_PHRASES):
        return True

    return False


def answer_question(question, db: Optional[FAISS] = None):
    """
    Priority:
      1. Uploaded documents (if a vector DB exists)
      2. Web search (if no DB, or documents can't answer)

    Returns (answer, sources, source_type)
      source_type: "document" | "web" | "none" | "error"
    """

    question = (question or "").strip()

    if not question:
        return "Please enter a question.", [], "none"

    try:
        # ---------- 1. DOCUMENTS FIRST ----------
        if db is not None:
            print("[ROUTER] documents available -> trying documents first")

            answer, docs = answer_from_documents(question, db)

            if not _is_not_found(answer):
                print("[ROUTER] answered from DOCUMENTS")
                return answer, docs, "document"

            print("[ROUTER] documents could not answer -> web fallback")
        else:
            print("[ROUTER] no documents uploaded -> web search")

        # ---------- 2. WEB FALLBACK ----------
        web_answer, web_sources = answer_from_web(question)

        if web_sources:
            print("[ROUTER] answered from WEB")
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
            filename = os.path.basename(source.metadata.get("source", "Unknown"))
            page = source.metadata.get("page", "?")
            print(f"- {filename} (Page {page})")

    elif source_type == "web":
        for index, source in enumerate(sources, start=1):
            print(f"- [{index}] {source.get('title', 'Unknown')}")
            print(f"  {source.get('url', '')}")

    elif source_type == "error":
        print("(Groq rate limit - see answer above)")

    print("=" * 60)