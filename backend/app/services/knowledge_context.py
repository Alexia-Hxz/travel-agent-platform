"""Retrieval glue between the vector store and the agent graphs.

Small on purpose. The graphs receive already-retrieved material in their initial state:
the services that build that state still hold a database session, whereas a graph node
runs on a worker thread and would have to open one of its own. Keeping the retrieval
here also means the planner and the adjustment agent share one definition of what
"the traveller's own documents" means.
"""

from app.services import vector_service


def retrieve_traveler_documents(
    trip_id: int, query: str, limit: int = vector_service.DEFAULT_RESULT_LIMIT
) -> tuple[list[dict], dict[str, str]]:
    """Return prompt material from the trip's uploaded documents, plus a timeline event.

    The event is what makes this observable: when nothing is retrieved the interface says
    whether the trip has no indexed documents, whether the query was empty, or whether
    the vector store itself was unreachable. An empty list on its own would say none of
    that, and the earlier retrieval attempt failed invisibly for exactly that reason.
    """
    outcome = vector_service.search_trip_documents(trip_id, query, limit)
    if outcome.status == "ok":
        sources = list(dict.fromkeys(chunk.source_label for chunk in outcome.chunks))
        return vector_service.document_context(outcome.chunks), {
            "tool": "document_search",
            "status": "completed",
            "message": f"已从上传资料中检索到 {len(outcome.chunks)} 段可用依据（来源：{'、'.join(sources)}）。",
        }
    return [], {
        "tool": "document_search",
        # "skipped" for a legitimately empty result, "fallback" when retrieval could not
        # run at all. Both leave the run working, but only one of them is a problem.
        "status": "skipped" if outcome.status in {"empty", "skipped"} else "fallback",
        "message": outcome.reason or "资料检索未完成，本次未使用上传资料。",
    }
