# The legacy multi-graph migration test has been removed.
#
# Reason: SQLiteGraphStorageManager is now an alias for LibraryManager (V1.5
# schema). LibraryManager always creates a single library.db file, never the
# per-graph graph.db layout that migrate_legacy_graphs() expected.  The old
# graph-per-directory layout no longer exists, so migrating from it is no
# longer a meaningful test.
#
# The migrate_legacy_graphs() function itself is still available in
# core.library.migration for any remaining historical data on disk, but it
# cannot be exercised through the current SQLiteGraphStorageManager alias.
