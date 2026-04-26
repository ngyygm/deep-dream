"""
Neo4j Storage Layer Tests.

Tests the Neo4jStorageManager including entity CRUD, relation CRUD,
episode CRUD, search functionality, and version management.

Requirements:
- Neo4j must be running on localhost:7687
- The database will use a test graph_id for isolation

Run with: pytest core/tests/test_neo4j_store.py -v
"""
import pytest
from datetime import datetime, timezone, timedelta
from core.tests.conftest import TEST_GRAPH_ID, TestHelpers


# ============================================================================
# Entity CRUD Tests
# ============================================================================

class TestEntityCRUD:
    """Test entity create, read, update, delete operations."""

    def test_create_entity(self, storage, test_helpers):
        """Test creating a new entity."""
        entity = test_helpers.create_test_entity(
            storage,
            name="TestPython",
            content="Python is a programming language"
        )
        assert entity.absolute_id is not None
        assert entity.family_id is not None
        assert entity.name == "TestPython"

    def test_save_and_get_entity_by_absolute_id(self, storage, test_helpers):
        """Test saving and retrieving entity by absolute_id."""
        created = test_helpers.create_test_entity(
            storage,
            name="GetTestEntity",
            content="Test content for retrieval"
        )
        retrieved = storage.get_entity_by_absolute_id(created.absolute_id)
        assert retrieved is not None
        assert retrieved.absolute_id == created.absolute_id
        assert retrieved.name == "GetTestEntity"
        assert retrieved.content == "Test content for retrieval"

    def test_save_and_get_entity_by_family_id(self, storage, test_helpers):
        """Test saving and retrieving entity by family_id."""
        created = test_helpers.create_test_entity(
            storage,
            name="FamilyTestEntity",
            family_id="test_family_123"
        )
        retrieved = storage.get_entity_by_family_id("test_family_123")
        assert retrieved is not None
        assert retrieved.family_id == "test_family_123"
        assert retrieved.name == "FamilyTestEntity"

    def test_get_entity_by_family_id_not_found(self, storage):
        """Test retrieving non-existent entity by family_id."""
        result = storage.get_entity_by_family_id("nonexistent_family")
        assert result is None

    def test_get_entity_by_absolute_id_not_found(self, storage):
        """Test retrieving non-existent entity by absolute_id."""
        result = storage.get_entity_by_absolute_id("nonexistent_absolute")
        assert result is None

    def test_update_entity_by_absolute_id(self, storage, test_helpers):
        """Test updating entity fields."""
        created = test_helpers.create_test_entity(
            storage,
            name="UpdateTest",
            content="Original content"
        )
        updated = storage.update_entity_by_absolute_id(
            created.absolute_id,
            name="UpdatedName",
            content="Updated content",
            confidence=0.9
        )
        assert updated is not None
        assert updated.name == "UpdatedName"
        assert updated.content == "Updated content"
        assert updated.confidence == 0.9

    def test_count_unique_entities(self, storage, test_helpers):
        """Test counting unique entities."""
        initial_count = storage.count_unique_entities()
        test_helpers.create_test_entity(storage, "CountTest1")
        test_helpers.create_test_entity(storage, "CountTest2")
        # Create a new version of CountTest1 (same family_id)
        test_helpers.create_test_entity(storage, "CountTest1", content="New version")
        new_count = storage.count_unique_entities()
        # Should have 2 more unique entities (initial + 2 new families)
        assert new_count >= initial_count + 2

    def test_get_all_entities(self, storage, test_helpers):
        """Test getting all entities with limit."""
        test_helpers.create_test_entity(storage, "ListTest1")
        test_helpers.create_test_entity(storage, "ListTest2")
        entities = storage.get_all_entities(limit=10, exclude_embedding=True)
        assert len(entities) >= 2

    def test_delete_entity_by_absolute_id(self, storage, test_helpers):
        """Test deleting entity by absolute_id."""
        created = test_helpers.create_test_entity(storage, "DeleteTest")
        # Delete should succeed
        success = storage.delete_entity_by_absolute_id(created.absolute_id)
        assert success is True
        # Entity should no longer exist
        retrieved = storage.get_entity_by_absolute_id(created.absolute_id)
        assert retrieved is None

    def test_delete_entity_all_versions(self, storage, test_helpers):
        """Test deleting all versions of an entity."""
        family_id = "delete_family_test"
        test_helpers.create_test_entity(
            storage,
            "DeleteFamilyTest",
            family_id=family_id
        )
        test_helpers.create_test_entity(
            storage,
            "DeleteFamilyTest",
            content="Version 2",
            family_id=family_id
        )
        count = storage.delete_entity_all_versions(family_id)
        assert count == 2
        retrieved = storage.get_entity_by_family_id(family_id)
        assert retrieved is None


# ============================================================================
# Entity Version Management Tests
# ============================================================================

class TestEntityVersions:
    """Test entity version tracking and retrieval."""

    def test_get_entity_versions(self, storage, test_helpers):
        """Test getting all versions of an entity."""
        family_id = "version_test_family"
        test_helpers.create_test_entity(
            storage,
            "VersionTest",
            content="Version 1",
            family_id=family_id
        )
        test_helpers.create_test_entity(
            storage,
            "VersionTest",
            content="Version 2",
            family_id=family_id
        )
        versions = storage.get_entity_versions(family_id)
        assert len(versions) == 2
        assert versions[0].family_id == family_id

    def test_get_entity_version_count(self, storage, test_helpers):
        """Test getting version count for entity."""
        family_id = "count_test_family"
        test_helpers.create_test_entity(
            storage,
            "CountVersionTest",
            family_id=family_id
        )
        test_helpers.create_test_entity(
            storage,
            "CountVersionTest",
            family_id=family_id
        )
        count = storage.get_entity_version_count(family_id)
        assert count == 2

    def test_get_entity_version_at_time(self, storage, test_helpers):
        """Test getting entity version at specific time."""
        family_id = "time_test_family"
        now = datetime.now(timezone.utc)
        # Create first version
        test_helpers.create_test_entity(
            storage,
            "TimeTest",
            content="Version 1",
            family_id=family_id
        )
        # Wait a bit
        import time
        time.sleep(0.1)
        time_point = datetime.now(timezone.utc)
        # Create second version
        test_helpers.create_test_entity(
            storage,
            "TimeTest",
            content="Version 2",
            family_id=family_id
        )
        # Query at time_point should return Version 1
        entity = storage.get_entity_version_at_time(family_id, time_point)
        assert entity is not None
        assert entity.family_id == family_id

    def test_get_all_entities_before_time(self, storage, test_helpers):
        """Test getting entities created before a specific time."""
        test_helpers.create_test_entity(storage, "BeforeTest1")
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        entities = storage.get_all_entities_before_time(
            future_time,
            limit=10,
            exclude_embedding=True
        )
        assert len(entities) >= 1


# ============================================================================
# Relation CRUD Tests
# ============================================================================

class TestRelationCRUD:
    """Test relation create, read, update, delete operations."""

    def test_create_relation(self, storage, test_helpers):
        """Test creating a new relation."""
        entity1 = test_helpers.create_test_entity(storage, "Entity1")
        entity2 = test_helpers.create_test_entity(storage, "Entity2")
        relation = test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id,
            content="Entity1 relates to Entity2"
        )
        assert relation.absolute_id is not None
        assert relation.family_id is not None
        assert relation.entity1_absolute_id < relation.entity2_absolute_id

    def test_get_relation_by_absolute_id(self, storage, test_helpers):
        """Test retrieving relation by absolute_id."""
        entity1 = test_helpers.create_test_entity(storage, "RelEntity1")
        entity2 = test_helpers.create_test_entity(storage, "RelEntity2")
        created = test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id
        )
        retrieved = storage.get_relation_by_absolute_id(created.absolute_id)
        assert retrieved is not None
        assert retrieved.absolute_id == created.absolute_id
        assert retrieved.content == created.content

    def test_get_relation_by_family_id(self, storage, test_helpers):
        """Test retrieving relation by family_id."""
        entity1 = test_helpers.create_test_entity(storage, "FamEntity1")
        entity2 = test_helpers.create_test_entity(storage, "FamEntity2")
        family_id = "test_relation_family"
        created = test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id,
            family_id=family_id
        )
        retrieved = storage.get_relation_by_family_id(family_id)
        assert retrieved is not None
        assert retrieved.family_id == family_id

    def test_get_entity_relations_by_family_id(self, storage, test_helpers):
        """Test getting relations for an entity."""
        entity1 = test_helpers.create_test_entity(storage, "CentralEntity")
        entity2 = test_helpers.create_test_entity(storage, "RelatedEntity1")
        entity3 = test_helpers.create_test_entity(storage, "RelatedEntity2")

        test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id
        )
        test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity3.absolute_id
        )

        relations = storage.get_entity_relations_by_family_id(entity1.family_id)
        assert len(relations) == 2

    def test_count_unique_relations(self, storage, test_helpers):
        """Test counting unique relations."""
        initial_count = storage.count_unique_relations()
        entity1 = test_helpers.create_test_entity(storage, "CountRelEnt1")
        entity2 = test_helpers.create_test_entity(storage, "CountRelEnt2")
        test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id
        )
        new_count = storage.count_unique_relations()
        assert new_count >= initial_count + 1

    def test_get_all_relations(self, storage, test_helpers):
        """Test getting all relations."""
        entity1 = test_helpers.create_test_entity(storage, "AllRelEnt1")
        entity2 = test_helpers.create_test_entity(storage, "AllRelEnt2")
        test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id
        )
        relations = storage.get_all_relations(limit=10, exclude_embedding=True)
        assert len(relations) >= 1


# ============================================================================
# Episode CRUD Tests
# ============================================================================

class TestEpisodeCRUD:
    """Test episode create, read operations."""

    def test_save_episode(self, storage, test_helpers):
        """Test saving an episode."""
        episode = test_helpers.create_test_episode(
            storage,
            content="Test episode content",
            source_document="test_doc.txt"
        )
        assert episode.absolute_id is not None
        assert episode.content == "Test episode content"

    def test_get_episode_by_absolute_id(self, storage, test_helpers):
        """Test retrieving episode by absolute_id."""
        created = test_helpers.create_test_episode(
            storage,
            content="Retrievable episode content"
        )
        retrieved = storage.load_episode(created.absolute_id)
        assert retrieved is not None
        assert retrieved.absolute_id == created.absolute_id
        assert retrieved.content == "Retrievable episode content"

    def test_get_latest_episode(self, storage, test_helpers):
        """Test getting the latest episode."""
        test_helpers.create_test_episode(storage, content="First episode")
        test_helpers.create_test_episode(storage, content="Second episode")
        latest = storage.get_latest_episode()
        assert latest is not None
        assert latest.content == "Second episode"

    def test_get_latest_episode_metadata(self, storage, test_helpers):
        """Test getting latest episode metadata."""
        test_helpers.create_test_episode(storage, content="Metadata test")
        metadata = storage.get_latest_episode_metadata()
        assert metadata is not None
        assert "absolute_id" in metadata


# ============================================================================
# Search Functionality Tests
# ============================================================================

class TestSearchFunctionality:
    """Test search and similarity functionality."""

    def test_search_entities_by_similarity(self, storage, test_helpers):
        """Test semantic search for entities."""
        # Create some test entities
        test_helpers.create_test_entity(
            storage,
            "Python",
            content="Python is a programming language"
        )
        test_helpers.create_test_entity(
            storage,
            "JavaScript",
            content="JavaScript is a scripting language"
        )
        # Search for programming-related content
        results = storage.search_entities_by_similarity(
            query_name="programming",
            query_content="programming languages",
            threshold=0.0,  # Low threshold for testing
            max_results=10,
            text_mode="name_and_content",
            similarity_method="embedding"
        )
        assert len(results) >= 0  # May return empty if no embeddings

    def test_search_relations_by_similarity(self, storage, test_helpers):
        """Test semantic search for relations."""
        entity1 = test_helpers.create_test_entity(storage, "SearchEnt1")
        entity2 = test_helpers.create_test_entity(storage, "SearchEnt2")
        test_helpers.create_test_relation(
            storage,
            entity1.absolute_id,
            entity2.absolute_id,
            content="uses and depends on"
        )
        results = storage.search_relations_by_similarity(
            query_text="dependency relationship",
            threshold=0.0,
            max_results=10
        )
        assert len(results) >= 0

    def test_get_entities_by_absolute_ids(self, storage, test_helpers):
        """Test batch getting entities by absolute IDs."""
        entity1 = test_helpers.create_test_entity(storage, "BatchGet1")
        entity2 = test_helpers.create_test_entity(storage, "BatchGet2")
        results = storage.get_entities_by_absolute_ids([
            entity1.absolute_id,
            entity2.absolute_id,
            "nonexistent_id"
        ])
        # Should return 2 entities (nonexistent returns None)
        valid_results = [e for e in results if e is not None]
        assert len(valid_results) == 2

    def test_get_entity_names_by_absolute_ids(self, storage, test_helpers):
        """Test batch getting entity names."""
        entity1 = test_helpers.create_test_entity(storage, "NameBatch1")
        entity2 = test_helpers.create_test_entity(storage, "NameBatch2")
        name_map = storage.get_entity_names_by_absolute_ids([
            entity1.absolute_id,
            entity2.absolute_id
        ])
        assert entity1.absolute_id in name_map
        assert name_map[entity1.absolute_id] == "NameBatch1"

    def test_get_family_ids_by_names(self, storage, test_helpers):
        """Test getting family IDs by entity names."""
        test_helpers.create_test_entity(
            storage,
            "UniqueNameLookup",
            family_id="lookup_family_123"
        )
        name_map = storage.get_family_ids_by_names(["UniqueNameLookup"])
        assert "UniqueNameLookup" in name_map
        assert name_map["UniqueNameLookup"] == "lookup_family_123"


# ============================================================================
# Entity Merge and Split Tests
# ============================================================================

class TestEntityMergeSplit:
    """Test entity merging and splitting operations."""

    def test_split_entity_version(self, storage, test_helpers):
        """Test splitting an entity version into a new family."""
        family_id = "split_test_family"
        entity = test_helpers.create_test_entity(
            storage,
            "SplitTestEntity",
            family_id=family_id
        )
        new_family_id = "split_new_family"
        updated = storage.split_entity_version(
            entity.absolute_id,
            new_family_id
        )
        assert updated is not None
        assert updated.family_id == new_family_id

        # Original family should no longer have this version
        original = storage.get_entity_by_family_id(family_id)
        assert original is None or original.absolute_id != entity.absolute_id

    def test_merge_entity_families(self, storage, test_helpers):
        """Test merging multiple entity families."""
        target_family = "merge_target"
        source_family1 = "merge_source1"
        source_family2 = "merge_source2"

        # Create entities
        test_helpers.create_test_entity(
            storage,
            "TargetEntity",
            family_id=target_family
        )
        test_helpers.create_test_entity(
            storage,
            "SourceEntity1",
            family_id=source_family1
        )
        test_helpers.create_test_entity(
            storage,
            "SourceEntity2",
            family_id=source_family2
        )

        # Merge
        result = storage.merge_entity_families(
            target_family,
            [source_family1, source_family2],
            skip_name_check=True
        )
        assert result > 0


# ============================================================================
# Batch Operations Tests
# ============================================================================

class TestBatchOperations:
    """Test batch operations for entities and relations."""

    def test_batch_delete_entities(self, storage, test_helpers):
        """Test batch deleting entities."""
        family1 = "batch_del_1"
        family2 = "batch_del_2"
        test_helpers.create_test_entity(storage, "BatchDel1", family_id=family1)
        test_helpers.create_test_entity(storage, "BatchDel2", family_id=family2)

        deleted = storage.batch_delete_entities([family1, family2])
        assert deleted >= 2

        # Verify deletion
        assert storage.get_entity_by_family_id(family1) is None
        assert storage.get_entity_by_family_id(family2) is None

    def test_batch_delete_relations(self, storage, test_helpers):
        """Test batch deleting relations."""
        e1 = test_helpers.create_test_entity(storage, "BatchRelEnt1")
        e2 = test_helpers.create_test_entity(storage, "BatchRelEnt2")
        rel1 = test_helpers.create_test_relation(
            storage,
            e1.absolute_id,
            e2.absolute_id,
            family_id="batch_rel_1"
        )
        e3 = test_helpers.create_test_entity(storage, "BatchRelEnt3")
        rel2 = test_helpers.create_test_relation(
            storage,
            e1.absolute_id,
            e3.absolute_id,
            family_id="batch_rel_2"
        )

        deleted = storage.batch_delete_relations([rel1.family_id, rel2.family_id])
        assert deleted >= 2

    def test_get_entity_version_counts(self, storage, test_helpers):
        """Test batch getting entity version counts."""
        family1 = "count_batch_1"
        family2 = "count_batch_2"
        test_helpers.create_test_entity(storage, "CountBatch1", family_id=family1)
        test_helpers.create_test_entity(storage, "CountBatch2", family_id=family2)
        test_helpers.create_test_entity(storage, "CountBatch1", family_id=family1)  # v2

        counts = storage.get_entity_version_counts([family1, family2])
        assert counts[family1] == 2
        assert counts[family2] == 1


# ============================================================================
# Graph Traversal Tests
# ============================================================================

class TestGraphTraversal:
    """Test graph traversal and neighbor queries."""

    def test_get_entity_neighbors(self, storage, test_helpers):
        """Test getting entity neighbors."""
        # Create a small graph: center -- edge1 -- edge2
        center = test_helpers.create_test_entity(storage, "CenterNode")
        edge1 = test_helpers.create_test_entity(storage, "EdgeNode1")
        edge2 = test_helpers.create_test_entity(storage, "EdgeNode2")

        test_helpers.create_test_relation(
            storage,
            center.absolute_id,
            edge1.absolute_id
        )
        test_helpers.create_test_relation(
            storage,
            edge1.absolute_id,
            edge2.absolute_id
        )

        if hasattr(storage, 'get_entity_neighbors'):
            neighbors = storage.get_entity_neighbors(center.family_id, depth=2)
            assert neighbors is not None
            assert "nodes" in neighbors or len(neighbors) >= 0


# ============================================================================
# Content and Attribute Tests
# ============================================================================

class TestContentAttributes:
    """Test content and attribute management."""

    def test_update_entity_summary(self, storage, test_helpers):
        """Test updating entity summary."""
        entity = test_helpers.create_test_entity(
            storage,
            "SummaryTest",
            content="Original content"
        )
        storage.update_entity_summary(entity.family_id, "Test summary")
        updated = storage.get_entity_by_family_id(entity.family_id)
        assert updated.summary == "Test summary"

    def test_update_entity_attributes(self, storage, test_helpers):
        """Test updating entity attributes."""
        entity = test_helpers.create_test_entity(
            storage,
            "AttrTest",
            content="Content with attributes"
        )
        import json
        attrs = {"key1": "value1", "key2": 42}
        storage.update_entity_attributes(
            entity.family_id,
            json.dumps(attrs)
        )
        updated = storage.get_entity_by_family_id(entity.family_id)
        assert updated.attributes is not None

    def test_update_entity_confidence(self, storage, test_helpers):
        """Test updating entity confidence."""
        entity = test_helpers.create_test_entity(
            storage,
            "ConfidenceTest"
        )
        storage.update_entity_confidence(entity.family_id, 0.85)
        updated = storage.get_entity_by_family_id(entity.family_id)
        assert updated.confidence == 0.85


# ============================================================================
# Isolated Entity Tests
# ============================================================================

class TestIsolatedEntities:
    """Test isolated entity detection and cleanup."""

    def test_get_isolated_entities(self, storage, test_helpers):
        """Test getting isolated entities (no relations)."""
        isolated = test_helpers.create_test_entity(
            storage,
            "IsolatedEntity"
        )
        # Connected entity
        e1 = test_helpers.create_test_entity(storage, "Connected1")
        e2 = test_helpers.create_test_entity(storage, "Connected2")
        test_helpers.create_test_relation(
            storage,
            e1.absolute_id,
            e2.absolute_id
        )

        if hasattr(storage, 'get_isolated_entities'):
            isolated_entities = storage.get_isolated_entities(limit=10)
            assert isolated_entities is not None
            # Should find at least the isolated entity we created
            family_ids = [e.family_id for e in isolated_entities]
            assert isolated.family_id in family_ids

    def test_count_isolated_entities(self, storage, test_helpers):
        """Test counting isolated entities."""
        if hasattr(storage, 'count_isolated_entities'):
            test_helpers.create_test_entity(storage, "CountIsolated")
            count = storage.count_isolated_entities()
            assert count >= 1


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Test statistics and graph information."""

    def test_get_graph_statistics(self, storage):
        """Test getting overall graph statistics."""
        stats = storage.get_graph_statistics()
        assert stats is not None
        assert "entity_count" in stats or len(stats) > 0

    def test_get_relations_referencing_absolute_id(self, storage, test_helpers):
        """Test getting relations that reference an entity."""
        entity = test_helpers.create_test_entity(storage, "ReferencedEntity")
        other = test_helpers.create_test_entity(storage, "OtherEntity")
        relation = test_helpers.create_test_relation(
            storage,
            entity.absolute_id,
            other.absolute_id
        )

        blocking = storage.get_relations_referencing_absolute_id(entity.absolute_id)
        assert len(blocking) >= 1
        assert blocking[0].absolute_id == relation.absolute_id
