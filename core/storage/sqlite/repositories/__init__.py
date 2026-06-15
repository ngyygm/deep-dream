"""SQLite repository package — thin SQL functions grouped by domain.

公共入口（Concept 统一原语）::

    from core.storage.sqlite.repositories.concepts import (
        upsert_concept_family, get_concept_family, find_concept_family,
        insert_concept_version, get_active_version,
        supersede_by_episodes, reactivate_by_episodes,
        list_concept_families, role_to_owner_type,
    )
"""

# 导出 Concept DAL（统一概念原语门面），保持模块内函数级导入以避免循环依赖。
from . import concepts  # noqa: F401

from .concepts import (  # noqa: F401
    ROLE_ENTITY,
    ROLE_RELATION,
    ROLE_EPISODE,
    ROLE_DOCUMENT,
    ALL_ROLES,
    DISPATCHABLE_ROLES,
    ROLE_TO_OWNER_TYPE,
    role_to_owner_type,
    upsert_concept_family,
    get_concept_family,
    find_concept_family,
    insert_concept_version,
    get_active_version,
    supersede_by_episodes,
    reactivate_by_episodes,
    list_concept_families,
)

__all__ = [
    "concepts",
    "ROLE_ENTITY",
    "ROLE_RELATION",
    "ROLE_EPISODE",
    "ROLE_DOCUMENT",
    "ALL_ROLES",
    "DISPATCHABLE_ROLES",
    "ROLE_TO_OWNER_TYPE",
    "role_to_owner_type",
    "upsert_concept_family",
    "get_concept_family",
    "find_concept_family",
    "insert_concept_version",
    "get_active_version",
    "supersede_by_episodes",
    "reactivate_by_episodes",
    "list_concept_families",
]
