"""Central configuration — env vars loaded once, paths pinned absolutely."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # why: keep env loading in one place

# --- repo-rooted paths (cwd-independent) -----------------------------------
# why: avoid the `app/data/chroma/` duplicate-DB trap when run from inside `app/`
_REPO_ROOT = Path(__file__).resolve().parent.parent

# --- OpenAI ----------------------------------------------------------------
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")

# Two model knobs so we can pay for quality only where it matters
# why: supervisor/grade are routing — small model fine; answer/reflect are user-facing — upgradeable
ROUTER_MODEL          = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
ANSWER_MODEL          = os.getenv("ANSWER_MODEL", "gpt-4.1")

# Legacy: keep LLM_MODEL working as a fallback for both
_LEGACY_LLM_MODEL     = os.getenv("LLM_MODEL", "")
if _LEGACY_LLM_MODEL:
    ROUTER_MODEL = ROUTER_MODEL or _LEGACY_LLM_MODEL
    ANSWER_MODEL = ANSWER_MODEL or _LEGACY_LLM_MODEL

EMBED_MODEL           = os.getenv("EMBED_MODEL", "text-embedding-3-large")
OPENAI_TIMEOUT_S      = int(os.getenv("OPENAI_TIMEOUT_S", "60"))

# --- Vector DB -------------------------------------------------------------
VECTOR_DB             = os.getenv("VECTOR_DB", "chroma")
CHROMA_DIR            = os.getenv("CHROMA_DIR", str(_REPO_ROOT / "data" / "chroma"))

# --- Reranker (optional) ---------------------------------------------------
# why: biggest quality lift per dollar; set RERANKER=cohere + COHERE_API_KEY to enable
RERANKER              = os.getenv("RERANKER", "").lower()       # "" (off) | "cohere"
COHERE_API_KEY        = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL   = os.getenv("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")

# --- Web search fallback ---------------------------------------------------
TAVILY_API_KEY        = os.getenv("TAVILY_API_KEY", "")

# --- Reflection gate -------------------------------------------------------
ENABLE_REFLECTION     = os.getenv("ENABLE_REFLECTION", "false").lower() == "true"

# --- /chat auth ------------------------------------------------------------
# why: prevents random scrapers from burning your OpenAI budget via the open /chat endpoint
CHAT_API_KEY          = os.getenv("CHAT_API_KEY", "")

# --- Logs ------------------------------------------------------------------
LOGS_DIR              = Path(os.getenv("LOGS_DIR", str(_REPO_ROOT / "logs")))

# --- Channel credentials (used by server.py webhooks) ----------------------
LINE_CHANNEL_SECRET   = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_TOKEN    = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
FB_PAGE_TOKEN         = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
FB_VERIFY_TOKEN       = os.getenv("FB_VERIFY_TOKEN", "change-me")
