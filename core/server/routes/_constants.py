"""Shared constants for all route modules."""

_BOOL_TRUE = frozenset(("1", "true", "yes", "on"))
_BOOL_FALSE = frozenset(("0", "false", "no", "off"))

_VALID_SEARCH_MODES = frozenset(("semantic", "bm25", "hybrid"))
_VALID_RERANKERS = frozenset(("rrf", "node_degree", "mmr"))
_VALID_TEXT_MODES = frozenset(("name_only", "content_only", "name_and_content"))
_VALID_SIM_METHODS = frozenset(("embedding", "text", "jaccard", "bleu"))
