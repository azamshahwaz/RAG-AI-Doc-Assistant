import os
import re
import sys
import json
import subprocess
import webbrowser
import datetime
import threading
import time
from rapidfuzz import fuzz
import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from faster_whisper import WhisperModel

# ============================================================
# WEB SEARCH BACKEND
#   pip install -U ddgs
# ============================================================

DDGS = None
_DDGS_ERRORS = (Exception,)

try:
    from ddgs import DDGS  # new package

    try:
        from ddgs.exceptions import (
            DDGSException,
            RatelimitException,
            TimeoutException,
        )
        _DDGS_ERRORS = (RatelimitException, TimeoutException, DDGSException)
    except Exception:
        pass

except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy fallback
        print("WARNING: using deprecated 'duckduckgo_search'. Run: pip install -U ddgs")
    except ImportError:
        print("ERROR: no search package installed. Run: pip install -U ddgs")

import rag_engine

# ============================================================
# CONVERSATION CONTEXT
# ============================================================

conversation_context = {
    "last_command": "",
    "last_intent": "",
    "last_topic": "",
    "last_website": "",
    "last_question": "",
}


def update_context(command, intent="", topic="", website=""):
    conversation_context["last_command"] = command

    if intent:
        conversation_context["last_intent"] = intent
    if topic:
        conversation_context["last_topic"] = topic
    if website:
        conversation_context["last_website"] = website

    conversation_context["last_question"] = command


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ============================================================
# HOSTED vs LOCAL DETECTION
# ============================================================
# On a hosted server (Streamlit Cloud / Render ...) the server
# is NOT the user's computer, so webbrowser.open() / subprocess
# open tabs on the server, never in the user's browser.
# There we send the URL to the user's browser instead.
#
# Override manually with an env var if detection is wrong:
#   HOSTED_ENV=1  -> force hosted mode
#   HOSTED_ENV=0  -> force local mode
# ============================================================


def is_hosted():
    forced = os.getenv("HOSTED_ENV")

    if forced == "1":
        return True
    if forced == "0":
        return False

    if (
        os.path.exists("/mount/src")            # Streamlit Cloud
        or os.getenv("RENDER")                  # Render
        or os.getenv("STREAMLIT_SHARING_MODE")
    ):
        return True

    if (
        sys.platform.startswith("linux")
        and not os.getenv("DISPLAY")
        and not os.getenv("WAYLAND_DISPLAY")
    ):
        return True

    return False


HOSTED = is_hosted()
print("Hosted mode:", HOSTED)

# ============================================================
# TEXT TO SPEECH - optional (not available on hosted servers)
# ============================================================

try:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)

except Exception as e:
    print("Local text-to-speech unavailable:", e)
    engine = None

# ============================================================
# WHISPER CONFIG
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

WHISPER_MODEL = "small"

START_TIMEOUT = 4.0
MAX_RECORD_SECONDS = 14.0

SILENCE_DURATION = 1.0
CHUNK_DURATION = 0.1

MIN_RMS_THRESHOLD = 0.004

PRE_ROLL_DURATION = 0.5

INITIAL_PROMPT = (
    "software development, AI, machine learning, "
    "RAG, documents, PDF, certifications, "
    "LangChain, FAISS, Python, Streamlit, "
    "Groq, embeddings, "
    "open YouTube, open Google, open Gmail, "
    "open GitHub, open LinkedIn, open Naukri"
)


def get_whisper_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    device, compute_type = get_whisper_device()

    print("=" * 60)
    print("Loading Whisper...")
    print("Model:", WHISPER_MODEL)
    print("Device:", device)
    print("Compute:", compute_type)
    print("=" * 60)

    try:
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=device, compute_type=compute_type
        )
    except Exception as e:
        print("Whisper GPU initialization failed:")
        print(e)
        print("Falling back to CPU...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )

    print("Whisper loaded successfully.")
    return _whisper_model


# ============================================================
# SPEAK
# ============================================================

def speak(text):
    if not text:
        return

    text = str(text)
    print("\nAssistant:", text)

    if engine is None:
        return

    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS Error:", e)


# ============================================================
# AUDIO HELPERS
# ============================================================

def calculate_rms(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(audio))))


def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    audio = audio - np.mean(audio)  # remove DC offset

    peak = np.max(np.abs(audio))

    if peak > 0.95:
        audio = (audio / peak) * 0.95

    return audio.astype(np.float32)


def calculate_noise_floor():
    if sd is None:
        return 0.01

    try:
        calibration_duration = 0.5

        print("Calibrating microphone... Please stay silent.")

        audio = sd.rec(
            int(calibration_duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        return calculate_rms(audio.flatten())

    except Exception as e:
        print("Noise calibration error:", e)
        return 0.01


# ============================================================
# RECORD SPEECH
# ============================================================

def record_speech():

    if sd is None:
        print("Microphone recording unavailable on this server")
        return None

    print("\nListening...")

    noise_floor = calculate_noise_floor()
    threshold = max(MIN_RMS_THRESHOLD, noise_floor * 2.0)

    print(f"Noise floor: {noise_floor:.5f}")
    print(f"Speech threshold: {threshold:.5f}")

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    max_chunks = int(MAX_RECORD_SECONDS / CHUNK_DURATION)
    timeout_chunks = int(START_TIMEOUT / CHUNK_DURATION)
    silence_chunks_required = int(SILENCE_DURATION / CHUNK_DURATION)
    pre_roll_chunks_required = max(1, int(PRE_ROLL_DURATION / CHUNK_DURATION))

    pre_roll = []
    audio_chunks = []

    speech_started = False
    silence_count = 0
    chunks_waited = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:

            for _ in range(max_chunks):

                data, overflowed = stream.read(chunk_samples)
                chunk = data[:, 0].copy()
                rms = calculate_rms(chunk)

                if overflowed:
                    print("Warning: microphone overflow.")

                if not speech_started:

                    chunks_waited += 1

                    pre_roll.append(chunk)
                    if len(pre_roll) > pre_roll_chunks_required:
                        pre_roll.pop(0)

                    if rms >= threshold:
                        speech_started = True
                        print("Speech detected.")
                        audio_chunks.extend(pre_roll)

                    elif chunks_waited >= timeout_chunks:
                        print("No speech detected.")
                        break

                else:
                    audio_chunks.append(chunk)

                    if rms < threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if silence_count >= silence_chunks_required:
                        print("Speech ended.")
                        break

    except Exception as e:
        print("Microphone error:", e)
        return None

    if not audio_chunks:
        return None

    audio = normalize_audio(np.concatenate(audio_chunks))
    duration = len(audio) / SAMPLE_RATE

    print(f"Recorded: {duration:.2f} seconds")

    if duration < 0.35:
        return None

    return audio


# ============================================================
# TRANSCRIPT CLEANING / VALIDATION
# ============================================================

def clean_transcript(text):
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?,])\1{2,}", r"\1", text)

    return text.strip()


def is_valid_transcript(text):
    if not text:
        return False

    text = text.strip()

    if len(text) < 2 or len(text) > 1000:
        return False

    words = text.lower().split()

    # Repeated word protection
    if len(words) >= 8:
        if len(set(words)) / len(words) < 0.25:
            return False

    # Repeated phrase protection
    if len(words) >= 12:
        for phrase_size in (2, 3, 4):
            counts = {}

            for i in range(len(words) - phrase_size + 1):
                phrase = " ".join(words[i:i + phrase_size])
                counts[phrase] = counts.get(phrase, 0) + 1

            if counts and max(counts.values()) >= 4:
                return False

    # Known Whisper hallucinations
    suspicious_phrases = (
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subscribe to my channel",
        "see you in the next video",
        "subtitles by",
        "amara.org",
    )

    if text.lower() in suspicious_phrases:
        return False

    return True


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(audio):
    if audio is None or len(audio) == 0:
        return ""

    model = get_whisper_model()

    try:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=3,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
            initial_prompt=INITIAL_PROMPT,
        )

        texts = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            if getattr(segment, "no_speech_prob", 0.0) >= 0.85:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -1.5:
                continue
            if getattr(segment, "compression_ratio", 0.0) > 3.0:
                continue

            texts.append(text)

        if not texts:
            return ""

        transcript = clean_transcript(" ".join(texts))

        if not is_valid_transcript(transcript):
            print("Suspicious transcript rejected:")
            print(transcript)
            return ""

        print("Detected language:", getattr(info, "language", "unknown"))
        print("You:", transcript)

        return transcript.lower()

    except Exception as e:
        print("Whisper error:", e)
        return ""


# ============================================================
# MAIN LISTEN FUNCTION
# ============================================================

def listen(duration=4, sample_rate=16000):
    try:
        audio = record_speech()

        if audio is None:
            print("No speech captured.")
            return ""

        transcript = transcribe_audio(audio)

        if not transcript:
            print("Could not understand speech.")
            return ""

        return transcript

    except KeyboardInterrupt:
        return ""

    except Exception as e:
        print("Listen error:", e)
        return ""


# ============================================================
# REMINDER
# ============================================================

def set_reminder(message, seconds):

    def reminder():
        time.sleep(seconds)
        speak(f"Reminder: {message}")

    threading.Thread(target=reminder, daemon=True).start()


# ============================================================
# OPEN URL  (LOCAL: server-side Chrome | HOSTED: user's browser)
# ============================================================

def _push_to_browser(url):
    """
    HOSTED MODE: open `url` in the USER's browser.

    1. Tries window.open() via a tiny hidden component. Browsers
       may block this (popup blocker) because a voice/text
       command is not a direct click.
    2. ALWAYS also shows a clickable button - a real user click
       is never blocked, so this is the guaranteed fallback.
    """

    try:
        import streamlit as st
        import streamlit.components.v1 as components
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            print("No Streamlit context - cannot send URL to browser.")
            return False

        st.session_state["pending_open_url"] = url

        components.html(
            f"<script>window.open({json.dumps(url)}, '_blank');</script>",
            height=0,
        )

        st.link_button(f"🌐 Open {url}", url)

        return True

    except Exception as e:
        print("Browser push failed:", e)
        return False


def render_pending_link():
    """
    OPTIONAL. Call this in app.py AFTER the chat/answer is drawn,
    only if the button disappears after a Streamlit rerun.
    """

    try:
        import streamlit as st

        url = st.session_state.get("pending_open_url")

        if url:
            st.link_button(f"🌐 Open {url}", url, key=f"open_{abs(hash(url))}")

    except Exception as e:
        print("render_pending_link error:", e)


def open_url_in_new_chrome_tab(url):
    """
    Opens `url` in a NEW tab.

    - Hosted server: sends the URL to the user's browser.
    - Local Windows: Chrome with --new-tab if installed.
    - Otherwise: default browser, new tab.
    """

    if HOSTED:
        return _push_to_browser(url)

    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
        ),
    ]

    chrome_path = next((p for p in chrome_paths if os.path.exists(p)), None)

    if chrome_path:
        try:
            subprocess.Popen([chrome_path, "--new-tab", url])
            return True
        except Exception as e:
            print("Chrome launch failed, using default browser:", e)

    webbrowser.open_new_tab(url)
    return True


# ============================================================
# SAFE WEB SEARCH
# ============================================================

SEARCH_BACKENDS = ["duckduckgo", "bing", "brave", "mojeek", "yahoo"]


def _normalise_result(result):
    """Different ddgs versions use different keys for the URL."""

    if not isinstance(result, dict):
        return None

    url = (
        result.get("href") or result.get("url") or result.get("link") or ""
    ).strip()

    title = (result.get("title") or result.get("name") or "").strip()

    body = (
        result.get("body")
        or result.get("snippet")
        or result.get("description")
        or ""
    ).strip()

    if not url or not url.startswith(("http://", "https://")):
        return None

    return {"title": title or url, "url": url, "body": body}


def web_search(query, max_results=5, region="in-en", retries=2):
    """Rate-limit tolerant search. Returns [] instead of raising."""

    if DDGS is None:
        print("Search unavailable: run  pip install -U ddgs")
        return []

    query = (query or "").strip()

    if not query:
        return []

    for backend in SEARCH_BACKENDS:

        for attempt in range(retries + 1):

            try:
                with DDGS() as ddgs:
                    try:
                        raw = ddgs.text(
                            query,
                            region=region,
                            safesearch="moderate",
                            max_results=max_results,
                            backend=backend,
                        )
                    except TypeError:
                        # Legacy duckduckgo_search has no `backend`
                        raw = ddgs.text(
                            query,
                            region=region,
                            safesearch="moderate",
                            max_results=max_results,
                        )

                    results = []
                    for item in (raw or []):
                        clean = _normalise_result(item)
                        if clean:
                            results.append(clean)

                if results:
                    print(f"Search OK via '{backend}': {len(results)} results")
                    return results

                print(f"Backend '{backend}' returned nothing (attempt {attempt + 1})")

            except _DDGS_ERRORS as e:
                print(f"Backend '{backend}' error (attempt {attempt + 1}): {e}")

            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

        time.sleep(0.5)

    print("All search backends failed for:", query)
    return []


# ============================================================
# KNOWN WEBSITES
# ============================================================

KNOWN_WEBSITES = {
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "linkedin": "https://www.linkedin.com",
    "hugging face": "https://huggingface.co",
    "nvidia": "https://www.nvidia.com",
    "microsoft": "https://www.microsoft.com",
    "stackoverflow": "https://stackoverflow.com",
    "tensorflow": "https://www.tensorflow.org",
    "python": "https://www.python.org",
    "mongodb": "https://www.mongodb.com",
    "oracle": "https://www.oracle.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "whatsapp": "https://web.whatsapp.com",
    "chatgpt": "https://chatgpt.com",
    "naukri": "https://www.naukri.com",
    "flipkart": "https://www.flipkart.com",
    "leetcode": "https://leetcode.com",
    "geeksforgeeks": "https://www.geeksforgeeks.org",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "spotify": "https://open.spotify.com",
}

# Matches: example.com / www.example.co.in / site.com/page
DOMAIN_PATTERN = re.compile(
    r"^(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/\S*)?$"
)


def resolve_website_name(name):
    name = name.lower().strip()

    best_match = None
    best_score = 0

    for website in KNOWN_WEBSITES:
        score = fuzz.ratio(name, website)

        if score > best_score:
            best_score = score
            best_match = website

    print(f"Website match: {name} -> {best_match} ({best_score})")

    if best_score >= 70:
        return best_match

    return name


# ============================================================
# SEARCH AND OPEN ANY WEBSITE
# ============================================================

def search_and_open_website(website_name, announce=True):
    """
    Resolution order:
      1. Full domain / URL  (example.com)   -> opened directly
      2. Known website map  (youtube, ...)  -> opened directly
      3. Anything else                      -> web search, best result

    Returns (success, message)
    """

    def respond(message, success):
        if announce and message:
            speak(message)
        return success, message

    def opened_message(name):
        if HOSTED:
            return (
                f"Opening {name} in a new tab. "
                f"Agar tab na khule, neeche wala button dabaiye."
            )
        return f"Opening {name} in a new tab."

    website_name = (website_name or "").strip().lower()

    if not website_name:
        return respond("Please tell me the website name.", False)

    # ---------------- DIRECT DOMAIN / URL ----------------

    if DOMAIN_PATTERN.match(website_name):

        url = website_name

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        print("\n" + "=" * 60)
        print("DOMAIN - OPENING DIRECTLY")
        print("URL:", url)
        print("=" * 60)

        open_url_in_new_chrome_tab(url)

        update_context(
            command=f"open {website_name}",
            intent="open_website",
            website=website_name,
        )

        return respond(opened_message(website_name), True)

    resolved_name = resolve_website_name(website_name)

    if not resolved_name:
        return respond("Please tell me the website name.", False)

    # ---------------- KNOWN WEBSITE ----------------

    if resolved_name in KNOWN_WEBSITES:

        selected_url = KNOWN_WEBSITES[resolved_name]

        print("\n" + "=" * 60)
        print("KNOWN WEBSITE - OPENING DIRECTLY")
        print("URL:", selected_url)
        print("=" * 60)

        open_url_in_new_chrome_tab(selected_url)

        update_context(
            command=f"open {resolved_name}",
            intent="open_website",
            website=resolved_name,
        )

        return respond(opened_message(resolved_name), True)

    # ---------------- UNKNOWN WEBSITE: SEARCH ----------------

    website_name = resolved_name

    print("\n" + "=" * 60)
    print("WEBSITE SEARCH")
    print("=" * 60)
    print("Searching for:", website_name)

    results = web_search(f"{website_name} official website", max_results=5)

    # Last resort: guess the domain
    if not results:

        guess = re.sub(r"[^a-z0-9]", "", website_name)

        if guess:
            guessed_url = f"https://www.{guess}.com"

            print("Search failed - trying guessed URL:", guessed_url)

            open_url_in_new_chrome_tab(guessed_url)

            update_context(
                command=f"open {website_name}",
                intent="open_website",
                website=website_name,
            )

            return respond(
                f"Search was unavailable, so I opened {guess} dot com in a new tab.",
                True,
            )

        return respond(f"I could not find the website {website_name}.", False)

    best = results[0]

    print("Website title:", best["title"])
    print("Website URL:", best["url"])

    open_url_in_new_chrome_tab(best["url"])

    update_context(
        command=f"open {website_name}",
        intent="open_website",
        website=website_name,
    )

    print("=" * 60)
    print("WEBSITE OPENED IN NEW TAB")
    print("=" * 60)

    return respond(opened_message(website_name), True)


# ============================================================
# UPLOADED DOCUMENT VECTORSTORE (cached, standalone CLI use)
# ============================================================

_cached_db = None
_db_loaded = False


def get_document_db():
    global _cached_db, _db_loaded

    if not _db_loaded:
        try:
            _cached_db = rag_engine.load_vectorstore()
        except Exception as e:
            print("Could not load saved vectorstore:", e)
            _cached_db = None

        _db_loaded = True

    return _cached_db


# ============================================================
# RAG + WEB ANSWER (documents first, web fallback, WITH sources)
# ============================================================

def describe_source(source_type):

    if source_type == "document":
        return "Yeh jankari maine aapke uploaded document se li hai."

    if source_type == "web":
        return "Yeh jankari maine web search se li hai."

    if source_type == "both":
        return "Yeh jawaab aapke document aur web search dono se mila hai."

    return ""


def format_document_sources(doc_sources, limit=None):
    """Lists every (already filtered) source doc. No fixed cap."""

    if not doc_sources:
        return ""

    seen = set()
    lines = []

    for doc in doc_sources:

        metadata = getattr(doc, "metadata", None) or {}
        filename = metadata.get("filename", metadata.get("source", "Unknown file"))
        page = metadata.get("page")
        slide = metadata.get("slide")
        sheet = metadata.get("sheet")

        if page is not None:
            try:
                location = f"Page {int(page) + 1}"
            except (ValueError, TypeError):
                location = f"Page {page}"
        elif slide is not None:
            location = f"Slide {slide}"
        elif sheet is not None:
            location = f"Sheet: {sheet}"
        else:
            location = "Document"

        key = (str(filename), location)

        if key in seen:
            continue

        seen.add(key)
        lines.append(f"- {os.path.basename(str(filename))} ({location})")

        if limit and len(lines) >= limit:
            break

    return "\n".join(lines)


def format_web_sources(web_sources, limit=3):

    if not web_sources:
        return ""

    lines = []

    for index, source in enumerate(web_sources[:limit], start=1):

        if not isinstance(source, dict):
            continue

        title = source.get("title", "Unknown source")

        url = source.get("url") or source.get("href") or source.get("link") or ""

        if url:
            lines.append(f"{index}. {title} - {url}")
        else:
            lines.append(f"{index}. {title}")

    return "\n".join(lines)


def voice_answer(question, db=None, context=None):
    """
    Single entry point used by BOTH the Streamlit app and the CLI.

    Returns: answer_text, reference_note, sources, source_type
    """

    try:
        answer, sources, source_type = rag_engine.answer_question(question, db)

    except rag_engine.RateLimitError as e:
        return (
            f"Groq ka daily token limit khatam ho gaya hai. {str(e)}",
            "",
            [],
            "error",
        )

    except Exception as e:
        return f"Sorry, ek error aa gaya: {e}", "", [], "error"

    reference_note = describe_source(source_type)

    if source_type == "document":

        docs_text = format_document_sources(sources)

        if docs_text:
            reference_note += "\n" + docs_text

    elif source_type == "web":

        web_text = format_web_sources(sources)

        if web_text:
            reference_note += "\n" + web_text

    elif source_type == "both" and isinstance(sources, dict):

        docs_text = format_document_sources(sources.get("documents", []))
        web_text = format_web_sources(sources.get("web", []))

        if docs_text:
            reference_note += "\nDocuments:\n" + docs_text
        if web_text:
            reference_note += "\nWeb:\n" + web_text

    return answer, reference_note.strip(), sources, source_type


# ============================================================
# "OPEN ..." COMMAND PARSING (English + Hinglish)
# ============================================================

OPEN_PREFIXES = ("open ", "launch ", "go to ", "goto ", "visit ")

OPEN_SUFFIXES = (
    " kholo",
    " khol do",
    " khol de",
    " kholiye",
    " khol dijiye",
    " open karo",
    " open kar do",
    " open kar de",
    " open kijiye",
)

MAX_OPEN_TARGET_WORDS = 4


def extract_open_target(command):
    """Returns the thing to open (e.g. "youtube"), or None."""

    target = None

    for prefix in OPEN_PREFIXES:
        if command.startswith(prefix):
            target = command[len(prefix):]
            break

    if target is None:
        for suffix in OPEN_SUFFIXES:
            if command.endswith(suffix):
                target = command[:-len(suffix)]
                break

    if target is None:
        return None

    # "... in a new tab" / "... on new chrome window"
    target = re.sub(
        r"\b(in|on|into)\s+(a\s+|the\s+)?new\s+(chrome\s+)?(tab|window)\b",
        " ",
        target,
    )

    # "... in chrome"
    target = re.sub(r"\b(in|on)\s+(google\s+)?chrome\b", " ", target)

    # "flipkart dot com" -> "flipkart.com"
    target = re.sub(r"\s+dot\s+", ".", target)

    target = re.sub(r"^(the|a|my)\s+", "", target.strip())
    target = re.sub(
        r"\s+(website|site|webpage|web page|please|for me|na|yaar)$",
        "",
        target.strip(),
    )

    return re.sub(r"\s+", " ", target).strip()


def _matches(command, exact=(), starts=()):
    if command in exact:
        return True

    return any(command.startswith(s) for s in starts)


# ============================================================
# SYSTEM COMMANDS
# ============================================================

def try_handle_system_command(command, announce=True):
    """
    Detect and execute a non-RAG "system" command: open app/website,
    time/date, reminder, explicit web search.

    Returns (handled, message)
      handled False -> caller should treat `command` as a RAG question.
    """

    def respond(message, handled=True):
        if announce and message:
            speak(message)
        return handled, message

    if not command:
        return False, ""

    command = command.strip().lower()

    # Remove dots only when NOT inside a domain (flipkart.com stays intact)
    command = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", command)
    command = re.sub(r"[!?,]+", " ", command)
    command = re.sub(r"\s+", " ", command).strip()

    print("\nProcessed command:", command)

    # ========================================================
    # OPEN APP / WEBSITE
    # ========================================================

    target = extract_open_target(command)

    if target:

        # ---------------- Desktop apps (local only) ----------------

        desktop_targets = (
            "chrome", "google chrome",
            "visual studio code", "vs code", "vscode", "code",
            "notepad",
            "calculator", "calc",
        )

        if target in desktop_targets:

            if HOSTED:

                if target in ("chrome", "google chrome"):
                    open_url_in_new_chrome_tab("https://www.google.com")
                    return respond(
                        "Opening Google in a new tab. "
                        "Agar tab na khule, neeche wala button dabaiye."
                    )

                return respond(
                    f"{target} desktop app hai, online version par open nahi ho sakta."
                )

            if target in ("chrome", "google chrome"):

                chrome_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    os.path.expandvars(
                        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
                    ),
                ]

                chrome_path = next(
                    (p for p in chrome_paths if os.path.exists(p)), None
                )

                if chrome_path:
                    subprocess.Popen(chrome_path)
                    return respond("Opening Google Chrome.")

                webbrowser.open_new_tab("https://www.google.com")
                return respond("Opening Chrome.")

            if target in ("visual studio code", "vs code", "vscode", "code"):

                code_path = (
                    r"C:\Users\HP\AppData\Local\Programs"
                    r"\Microsoft VS Code\Code.exe"
                )

                if os.path.exists(code_path):
                    subprocess.Popen(code_path)
                    return respond("Opening Visual Studio Code.")

                return respond("Visual Studio Code was not found.")

            if target == "notepad":
                subprocess.Popen("notepad.exe")
                return respond("Opening Notepad.")

            if target in ("calculator", "calc"):
                subprocess.Popen("calc.exe")
                return respond("Opening Calculator.")

        # ---------------- Any website -> NEW TAB ----------------
        # Long sentences are questions, not website names.

        if len(target.split()) <= MAX_OPEN_TARGET_WORDS:

            print("Website command detected:", target)

            _success, message = search_and_open_website(target, announce=False)

            return respond(message)

    # ========================================================
    # TIME
    # ========================================================

    if _matches(
        command,
        exact=(
            "time", "current time", "the time", "time batao",
            "samay batao", "abhi kya time hua hai",
        ),
        starts=(
            "what time is it", "what is the time", "what's the time",
            "whats the time", "tell me the time",
            "what is the current time", "what's the current time",
        ),
    ):
        now = datetime.datetime.now().strftime("%I:%M %p")
        update_context(command, intent="time")
        return respond(f"The time is {now}")

    # ========================================================
    # DATE
    # ========================================================

    if _matches(
        command,
        exact=(
            "date", "today's date", "todays date", "today date",
            "aaj ki date batao", "aaj ki tarikh batao",
        ),
        starts=(
            "what date is it", "what is the date", "what's the date",
            "whats the date", "what is today's date", "what's today's date",
            "what is todays date", "tell me the date", "tell me today's date",
        ),
    ):
        today = datetime.datetime.now().strftime("%d %B %Y")
        update_context(command, intent="date")
        return respond(f"Today is {today}")

    # ========================================================
    # REMINDER
    # ========================================================

    if command.startswith("remind me to"):

        try:
            task = command.split("remind me to", 1)[1].strip()

            if " in " in task and "minute" in task:

                message = task.split(" in ", 1)[0].strip()
                minutes = int(task.split(" in ", 1)[1].split()[0])

                update_context(command, intent="reminder")
                set_reminder(message, minutes * 60)

                return respond(
                    f"Reminder set. I will remind you to {message} "
                    f"in {minutes} minutes."
                )

            if " in " in task and "hour" in task:

                message = task.split(" in ", 1)[0].strip()
                hours = int(task.split(" in ", 1)[1].split()[0])

                update_context(command, intent="reminder")
                set_reminder(message, hours * 3600)

                return respond(
                    f"Reminder set. I will remind you to {message} "
                    f"in {hours} hours."
                )

            return respond("Please specify the reminder time in minutes or hours.")

        except Exception as e:
            print("Reminder error:", e)
            return respond("I could not understand the reminder time.")

    # ========================================================
    # EXPLICIT WEB SEARCH  (also Hinglish: "google karo ...")
    # ========================================================

    SEARCH_PREFIXES = (
        "search ",
        "search for ",
        "google ",
        "web search ",
        "search karo ",
        "google karo ",
    )

    matched_prefix = next(
        (p for p in SEARCH_PREFIXES if command.startswith(p)), None
    )

    if matched_prefix:

        query = command[len(matched_prefix):].strip()

        query = re.sub(r"\s+(karo|kar do|kijiye|please)$", "", query).strip()

        if not query:
            return respond("What should I search for?")

        update_context(command, intent="web_search", topic=query)

        answer = None
        web_sources = []

        try:
            answer, web_sources = rag_engine.answer_from_web(query)

        except rag_engine.RateLimitError as e:
            return respond(f"Groq ka daily token limit khatam ho gaya hai. {e}")

        except Exception as e:
            print("rag_engine web search failed:", e)
            answer = None

        # Fallback: raw results from our own searcher
        if not answer:

            web_sources = web_search(query, max_results=5)

            if not web_sources:
                return respond(
                    "Web search abhi available nahi hai. "
                    "Thodi der baad try kijiye."
                )

            answer = web_sources[0].get("body") or "Yeh top results mile:"

        sources_text = format_web_sources(web_sources)

        full_message = answer

        if sources_text:
            full_message += "\n\n" + sources_text

        return respond(full_message)

    # ========================================================
    # NOT A SYSTEM COMMAND - let the caller run RAG instead
    # ========================================================

    return False, ""


# ============================================================
# PROCESS COMMAND (standalone CLI loop)
# ============================================================

def process_command(command):

    if not command:
        return

    handled, _message = try_handle_system_command(command, announce=True)

    if handled:
        return

    db = get_document_db()

    answer, reference_note, _sources, source_type = voice_answer(
        command, db, conversation_context
    )

    update_context(command=command, intent="question")

    speak(answer)

    if source_type != "error" and reference_note:
        print("\n" + reference_note)


# ============================================================
# TESTS
# ============================================================

def test_microphone():

    print("=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)

    audio = record_speech()

    if audio is None:
        print("No audio captured.")
        return

    text = transcribe_audio(audio)

    print("=" * 60)
    print("FINAL TRANSCRIPT")
    print("=" * 60)
    print(text)


def test_search(query="python official website"):

    print("=" * 60)
    print("WEB SEARCH TEST")
    print("Query:", query)
    print("=" * 60)

    results = web_search(query, max_results=5)

    if not results:
        print("NO RESULTS - every backend failed or is rate limited.")
        return

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']}")
        print("   ", result["url"])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    speak("Hello, I am your AI Research Assistant. Ask me anything.")

    while True:

        question = listen()

        if not question:
            speak("Sorry, I could not understand you.")
            continue

        if question in ["exit", "quit", "stop", "goodbye", "close assistant"]:
            speak("Goodbye. Have a great day.")
            break

        process_command(question)