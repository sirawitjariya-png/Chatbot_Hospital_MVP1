"""Write per-user conversation traces.

Two outputs (both run on every call — no flag):
  1. logs/<user_id>/YYYY-MM.log  → pretty human-readable file (for local debugging)
  2. logging.info(json.dumps(...))  → single-line JSON to stdout (Cloud Run auto-ingests to Cloud Logging)

The file write is best-effort: if the runtime filesystem is read-only or the
path can't be created (e.g. on a stricter Cloud Run config), we just skip it
and keep the JSON log so production debugging still works.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import LOGS_DIR

log = logging.getLogger("trace")
W = 72

_STAGE_LABEL = {"md": "markdown", "url": "urls", "kb": "knowledge base", "web": "web search", "": ""}


def _box(label: str) -> str:
    return f"  ┌─ {label}"


def _lines(text: str, indent: str = "  │  ") -> list[str]:
    return [f"{indent}{ln}" for ln in text.splitlines()]


def _render_pretty(user_id: str, question: str, trace: list, final_answer: str, ts: str) -> str:
    out: list[str] = []
    out.append("=" * W)
    out.append(f"  {ts}   user: {user_id}")
    out.append("=" * W)
    out.append("")
    out.append(_box("QUESTION"))
    out += _lines(question)
    out.append("")

    for entry in trace:
        node = entry.get("node", "?")

        if node == "supervisor":
            out.append(_box(f"ROUTING  →  {entry['route'].upper()}"))
            out.append("")

        elif node == "retrieve":
            n = entry.get("chunks", 0)
            status = f"{n} chunk(s) found" if n else "nothing found"
            out.append(_box(f"RETRIEVE (knowledge base)  —  {status}"))
            sources = entry.get("sources") or []
            texts = entry.get("texts") or []
            for i, text in enumerate(texts, 1):
                src = sources[i - 1] if i - 1 < len(sources) else ""
                out.append(f"  │  ── chunk {i} ({src}) {'─' * (W - 22 - len(src))}")
                out += _lines(text)
            out.append("")

        elif node == "grade_chunks":
            status = "✓ relevant" if entry.get("relevant") else "✗ not relevant — trying web search"
            out.append(_box(f"GRADE CHUNKS  →  {status}"))
            out.append("")

        elif node == "web_search":
            out.append(_box("WEB SEARCH  (Tavily fallback)"))
            out.append(f"  │  query   : {entry.get('query', '')}")
            out.append(f"  │  results : {entry.get('results_count', 0)} snippet(s)")
            out.append("")

        elif node == "answer":
            stage = _STAGE_LABEL.get(entry.get("stage", ""), "")
            heading = f"DRAFT ANSWER  ({stage})" if stage else "DRAFT ANSWER"
            out.append(_box(heading))
            out += _lines(entry["draft"])
            out.append("")

        elif node == "reflect":
            status = "✎ Revised" if entry.get("changed") else "✓ Accepted as-is"
            out.append(_box(f"REFLECTION  ({status})"))
            out.append("")

        elif node == "no_data":
            out.append(_box("NO DATA  —  all sources exhausted"))
            out.append("")

        elif node == "smalltalk":
            out.append(_box("SMALLTALK  (no retrieval needed)"))
            out.append("")

        elif node == "off_topic":
            out.append(_box("OFF-TOPIC  (question not hospital-related)"))
            out.append("")

    out.append(_box("FINAL ANSWER"))
    out += _lines(final_answer)
    out.append("")
    out.append("-" * W)
    out.append("")
    return "\n".join(out) + "\n"


def write_trace(user_id: str, question: str, trace: list, final_answer: str) -> None:
    uid = user_id or "unknown"
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---- 1. Pretty local file (best-effort) ---------------------------------
    try:
        user_dir = LOGS_DIR / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        log_file = user_dir / f"{now.strftime('%Y-%m')}.log"
        pretty = _render_pretty(uid, question, trace, final_answer, ts)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(pretty)
    except Exception as e:
        log.warning("local trace write failed (%s) — continuing with JSON log only", e)

    # ---- 2. Structured JSON to stdout for Cloud Logging ---------------------
    record = {
        "type": "trace",
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": uid,
        "question": question,
        "trace": trace,
        "final_answer": final_answer,
    }
    try:
        log.info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        # never raise from tracer
        log.warning("json trace emit failed: %s", e)
