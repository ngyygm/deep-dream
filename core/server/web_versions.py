"""
Extracted from web.py — Version-related route handlers for GraphWebServer.
"""
from __future__ import annotations

import logging
from datetime import datetime

from flask import jsonify, request

logger = logging.getLogger(__name__)


def register_version_routes(server):
    """Register entity/relation version routes on *server.app*."""

    @server.app.route('/api/entities/<family_id>/versions')
    def get_entity_versions(family_id):
        """Get all versions of an entity."""
        try:
            versions = server.storage.get_entity_versions(family_id)

            if not versions:
                return jsonify({
                    'success': False,
                    'error': f'未找到实体 {family_id} 的版本'
                }), 404

            versions_data = []
            for i, entity in enumerate(versions, 1):
                versions_data.append({
                    'index': i,
                    'total': len(versions),
                    'absolute_id': entity.absolute_id,
                    'family_id': entity.family_id,
                    'name': entity.name,
                    'content': entity.content,
                    'event_time': entity.event_time.isoformat() if entity.event_time else None,
                    'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                    'episode_id': entity.episode_id
                })

            return jsonify({
                'success': True,
                'family_id': family_id,
                'versions': versions_data
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @server.app.route('/api/entities/<family_id>/versions/<absolute_id>')
    def get_entity_version(family_id, absolute_id):
        """Get a specific entity version."""
        try:
            entity = server.storage.get_entity_by_absolute_id(absolute_id)

            if not entity:
                return jsonify({
                    'success': False,
                    'error': f'未找到实体版本 {absolute_id}'
                }), 404

            if entity.family_id != family_id:
                return jsonify({
                    'success': False,
                    'error': f'实体ID不匹配'
                }), 400

            # Get version index
            versions = server.storage.get_entity_versions(family_id)
            version_index = next((i for i, e in enumerate(versions, 1) if e.absolute_id == absolute_id), None)

            # Get embedding preview (first 4 values)
            embedding_preview = server.storage.get_entity_embedding_preview(absolute_id, 4)

            # Get episode md content and json original text
            episode_content = None
            episode_text = None
            source_document = None
            doc_name = None
            if entity.episode_id:
                episode = server.storage.load_episode(entity.episode_id)
                if episode:
                    episode_content = episode.content
                    source_document = getattr(episode, 'source_document', None) or getattr(episode, 'doc_name', None)
                episode_text = server.storage.get_episode_text(entity.episode_id)

            return jsonify({
                'success': True,
                'entity': {
                    'absolute_id': entity.absolute_id,
                    'family_id': entity.family_id,
                    'name': entity.name,
                    'content': entity.content,
                    'event_time': entity.event_time.isoformat() if entity.event_time else None,
                    'processed_time': entity.processed_time.isoformat() if entity.processed_time else None,
                    'episode_id': entity.episode_id,
                    'episode_content': episode_content,
                    'episode_text': episode_text,
                    'source_document': source_document,
                    'doc_name': source_document,
                    'version_index': version_index,
                    'total_versions': len(versions),
                    'embedding_preview': embedding_preview
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @server.app.route('/api/relations/<family_id>/versions')
    def get_relation_versions(family_id):
        """Get all versions of a relation."""
        try:
            versions = server.storage.get_relation_versions(family_id)

            if not versions:
                return jsonify({
                    'success': False,
                    'error': f'未找到关系 {family_id} 的版本'
                }), 404

            # Batch-get all referenced entities
            all_abs_ids = set()
            for rel in versions:
                all_abs_ids.add(rel.entity1_absolute_id)
                all_abs_ids.add(rel.entity2_absolute_id)
            entity_map = server.storage.get_entities_by_absolute_ids(list(all_abs_ids)) if all_abs_ids else {}

            versions_data = []
            for i, relation in enumerate(versions, 1):
                entity1 = entity_map.get(relation.entity1_absolute_id)
                entity2 = entity_map.get(relation.entity2_absolute_id)

                versions_data.append({
                    'index': i,
                    'total': len(versions),
                    'absolute_id': relation.absolute_id,
                    'family_id': relation.family_id,
                    'content': relation.content,
                    'event_time': relation.event_time.isoformat() if relation.event_time else None,
                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                    'episode_id': relation.episode_id,
                    'entity1_absolute_id': relation.entity1_absolute_id,
                    'entity2_absolute_id': relation.entity2_absolute_id,
                    'entity1_id': entity1.family_id if entity1 else None,
                    'entity2_id': entity2.family_id if entity2 else None
                })

            return jsonify({
                'success': True,
                'family_id': family_id,
                'versions': versions_data
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @server.app.route('/api/relations/<family_id>/versions/<absolute_id>')
    def get_relation_version(family_id, absolute_id):
        """Get a specific relation version."""
        try:
            versions = server.storage.get_relation_versions(family_id)
            relation = next((r for r in versions if r.absolute_id == absolute_id), None)

            if not relation:
                return jsonify({
                    'success': False,
                    'error': f'未找到关系版本 {absolute_id}'
                }), 404

            if relation.family_id != family_id:
                return jsonify({
                    'success': False,
                    'error': f'关系ID不匹配'
                }), 400

            entity1 = server.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
            entity2 = server.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)

            version_index = next((i for i, r in enumerate(versions, 1) if r.absolute_id == absolute_id), None)

            embedding_preview = server.storage.get_relation_embedding_preview(absolute_id, 4)

            return jsonify({
                'success': True,
                'relation': {
                    'absolute_id': relation.absolute_id,
                    'family_id': relation.family_id,
                    'content': relation.content,
                    'event_time': relation.event_time.isoformat() if relation.event_time else None,
                    'processed_time': relation.processed_time.isoformat() if relation.processed_time else None,
                    'episode_id': relation.episode_id,
                    'entity1_absolute_id': relation.entity1_absolute_id,
                    'entity2_absolute_id': relation.entity2_absolute_id,
                    'entity1_id': entity1.family_id if entity1 else None,
                    'entity2_id': entity2.family_id if entity2 else None,
                    'entity1_name': entity1.name if entity1 else None,
                    'entity2_name': entity2.name if entity2 else None,
                    'version_index': version_index,
                    'total_versions': len(versions),
                    'embedding_preview': embedding_preview
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
