"""Neo4j EpisodeStoreMixin — composition of specialised sub-mixins."""
from ._episode_cache import EpisodeCacheMixin
from ._episode_mutations import EpisodeMutationMixin
from ._episode_queries import EpisodeQueryMixin


class EpisodeStoreMixin(EpisodeCacheMixin, EpisodeQueryMixin, EpisodeMutationMixin):
    """Combined episode store mixin — delegates to specialised sub-mixins.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
        self._episode_write_lock     → threading.Lock for episode writes
        self.cache_dir               → Path to episode cache dir
        self.cache_json_dir          → Path to JSON cache dir
        self.cache_md_dir            → Path to MD cache dir
        self.docs_dir                → Path to docs dir
        self._id_to_doc_hash         → Dict mapping cache_id to doc_hash
    """
    pass
