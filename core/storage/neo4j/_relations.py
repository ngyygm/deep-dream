"""Neo4j RelationStoreMixin — composition module delegating to specialized sub-mixins."""
from ._relation_embedding import RelationEmbeddingMixin
from ._relation_mutations import RelationMutationMixin
from ._relation_queries import RelationQueryMixin


class RelationStoreMixin(RelationEmbeddingMixin, RelationQueryMixin, RelationMutationMixin):
    """Combined relation store mixin — delegates to specialized sub-mixins."""
    pass
