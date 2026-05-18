"""SQLiteGraphStorageManager — composes all storage mixins.

Drop-in replacement for Neo4jStorageManager. Implements all methods across
10 Neo4j mixin interfaces (Base, Entity, Relation, Episode, Search, Stats,
GraphTraversal, Community, Dream, Concepts) using SQLite + FTS5 + numpy
brute-force cosine similarity.
"""

from ._base import _BaseMixin
from ._entity_save import _EntitySaveMixin
from ._entity_read import _EntityReadMixin
from ._entity_write import _EntityWriteMixin
from ._entity_util import _EntityUtilMixin
from ._stats import _StatsMixin
from ._relation_save import _RelationSaveMixin
from ._relation_read import _RelationReadMixin
from ._relation_write import _RelationWriteMixin
from ._episode import _EpisodeMixin
from ._search import _SearchMixin
from ._traversal import _TraversalMixin
from ._community import _CommunityMixin
from ._dream import _DreamMixin
from ._concepts import _ConceptMixin
from ._snapshot import _SnapshotMixin


class SQLiteGraphStorageManager(
    _BaseMixin,
    _EntitySaveMixin,
    _EntityReadMixin,
    _EntityWriteMixin,
    _EntityUtilMixin,
    _StatsMixin,
    _RelationSaveMixin,
    _RelationReadMixin,
    _RelationWriteMixin,
    _EpisodeMixin,
    _SearchMixin,
    _TraversalMixin,
    _CommunityMixin,
    _DreamMixin,
    _ConceptMixin,
    _SnapshotMixin,
):
    """SQLite graph storage — all methods from mixins."""
    pass
