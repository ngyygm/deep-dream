"""Neo4j EntityStoreMixin — composition module that delegates to specialized sub-mixins.

The original monolithic ~2200-line file has been split into:
  - _entity_embedding.py   (embedding computation, cache management, invalidation)
  - _entity_confidence.py  (confidence adjustment on contradiction/corroboration)
  - _entity_queries.py     (all read-only entity query methods)
  - _entity_mutations.py   (all entity write/mutation methods)

This file re-exports EntityStoreMixin via multiple inheritance so the import path
`from ._entities import EntityStoreMixin` continues to work unchanged.
"""

from ._entity_embedding import EntityEmbeddingMixin
from ._entity_confidence import EntityConfidenceMixin
from ._entity_queries import EntityQueryMixin
from ._entity_mutations import EntityMutationMixin


class EntityStoreMixin(
    EntityEmbeddingMixin,
    EntityConfidenceMixin,
    EntityQueryMixin,
    EntityMutationMixin,
):
    """Combined entity store mixin — delegates to specialized sub-mixins.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._write_lock             -> threading.Lock for entity writes
        self._cache                  -> QueryCache
        self.embedding_client        -> EmbeddingClient (optional)
        self._entity_emb_cache       -> embedding cache list
        self._entity_emb_cache_ts    -> embedding cache timestamp
        self._emb_cache_ttl          -> cache TTL in seconds
        self.entity_content_snippet_length -> content snippet length
    """

    pass
