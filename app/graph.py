"""CRAG-style LangGraph workflow:

    supervisor → smalltalk / off_topic → END
              ↘ retrieve → grade_chunks →
                    relevant     → answer → (reflect?) → END
                    not relevant → web_search → answer_web → (reflect?) → END

Worst case: 3–4 LLM calls (supervisor + grade + answer + maybe reflect).
Old graph worst-case was ~10.
"""
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import (
    supervisor,
    grade_chunks,
    draft_answer,
    reflect,
    web_search,
    no_data,
    smalltalk,
    off_topic,
)
from .rag import retrieve
from .config import ENABLE_REFLECTION
from .tracer import write_trace

log = logging.getLogger(__name__)

_MAX_HISTORY = 10
_user_history: dict[str, list] = {}  # user_id → last ≤10 messages
# NOTE: this is in-memory; on Cloud Run it does NOT survive cold starts and
# does NOT share across instances. Move to Firestore/Redis for production scale.


class State(TypedDict, total=False):
    question: str
    user_id: str
    route: str
    stage: str
    retrieved: list           # [{text, source, source_type}, ...]
    context: str
    chunks_relevant: bool
    draft: str
    answer: str
    history: list
    trace: Annotated[list, operator.add]


# --------------------------- nodes -------------------------------------------
def _retrieve_node(state):
    """Single retrieve from the unified collection (md + url merged)."""
    hits = retrieve(state["question"], k=5)
    context_text = "\n\n".join(h["text"] for h in hits) if hits else "(no relevant context found)"
    return {
        "retrieved": hits,
        "context": context_text,
        "stage": "kb",
        "trace": [{
            "node": "retrieve",
            "chunks": len(hits),
            "texts": [h["text"] for h in hits],
            "sources": [h.get("source_type", "") for h in hits],
        }],
    }


def _route_after_grade(state) -> str:
    return "ok" if state.get("chunks_relevant") else "web"


# --------------------------- graph build -------------------------------------
def build_graph():
    g = StateGraph(State)

    g.add_node("supervisor",   supervisor)
    g.add_node("retrieve",     _retrieve_node)
    g.add_node("grade_chunks", grade_chunks)
    g.add_node("draft_answer",  draft_answer)
    g.add_node("web_search",    web_search)
    g.add_node("answer_web",    draft_answer)
    g.add_node("reflect",      reflect)
    g.add_node("no_data",      no_data)
    g.add_node("smalltalk",    smalltalk)
    g.add_node("off_topic",    off_topic)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["route"],
        {"rag": "retrieve", "smalltalk": "smalltalk", "off_topic": "off_topic"},
    )

    # RAG path
    g.add_edge("retrieve", "grade_chunks")
    g.add_conditional_edges(
        "grade_chunks",
        _route_after_grade,
        {"ok": "draft_answer", "web": "web_search"},
    )

    # Web fallback path
    g.add_edge("web_search", "answer_web")

    # Reflection gate — why: skip an extra LLM call when ENABLE_REFLECTION=false
    if ENABLE_REFLECTION:
        g.add_edge("draft_answer", "reflect")
        g.add_edge("answer_web",   "reflect")
        g.add_edge("reflect",      END)
    else:
        g.add_edge("draft_answer", END)
        g.add_edge("answer_web",   END)

    g.add_edge("smalltalk", END)
    g.add_edge("off_topic", END)
    g.add_edge("no_data",   END)
    return g.compile()


_graph = build_graph()


_FALLBACK = (
    "ขออภัย ระบบเกิดข้อผิดพลาดชั่วคราว กรุณาลองใหม่อีกครั้ง "
    "หรือติดต่อโรงพยาบาลวลัยลักษณ์โดยตรงค่ะ\n\n"
    "Sorry, a temporary error occurred. Please try again or contact "
    "Walailuk Hospital directly."
)


def ask(question: str, user_id: str = "cli") -> str:
    history = _user_history.get(user_id, [])
    try:
        result = _graph.invoke({
            "question": question,
            "user_id": user_id,
            "history": history,
            "trace": [],
        })
        answer_text = result["answer"]
        _user_history[user_id] = (history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_text},
        ])[-_MAX_HISTORY:]
        try:
            write_trace(
                user_id=result.get("user_id", user_id),
                question=question,
                trace=result.get("trace", []),
                final_answer=answer_text,
            )
        except Exception as e:
            log.warning("write_trace failed: %s", e)
        return answer_text
    except Exception as e:
        log.error("Graph error for user %s: %s", user_id, e)
        return _FALLBACK
