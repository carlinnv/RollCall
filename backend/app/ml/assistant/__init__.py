"""F7 counselor assistant (PRD 2 section 3.4): retrieval, benchmark and a read-only Deep Agents agent.

The agent module imports deepagents / langchain lazily, so retrieval and the benchmark work
without them.
"""
from .corpus import AssistantData, Message, build_corpus, message_id_for, week_days, week_of
from .retrieval import MessageIndex, ParsedQuery, SearchResult, parse_query

__all__ = ["AssistantData", "Message", "MessageIndex", "ParsedQuery", "SearchResult", "build_corpus",
           "message_id_for", "parse_query", "week_days", "week_of"]
