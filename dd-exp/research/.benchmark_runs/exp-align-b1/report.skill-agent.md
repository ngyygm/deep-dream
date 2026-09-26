# longmemeval-s benchmark report

- Total: 3
- Scored: 3
- Overall: 1.0000

## Scores by type

| Type | Score | Count |
|---|---:|---:|
| temporal-reasoning | 1.0000 | 3 |

## Retrieval

| Metric | Score |
|---|---:|
| session_evidence_recall@1 | 0.6667 |
| session_evidence_recall@10 | 1.0000 |
| session_evidence_recall@3 | 1.0000 |
| session_evidence_recall@30 | 1.0000 |
| session_evidence_recall@5 | 1.0000 |
| session_evidence_recall@50 | 1.0000 |
| session_ndcg_any@1 | 1.0000 |
| session_ndcg_any@10 | 1.0000 |
| session_ndcg_any@3 | 1.0000 |
| session_ndcg_any@30 | 1.0000 |
| session_ndcg_any@5 | 1.0000 |
| session_ndcg_any@50 | 1.0000 |
| session_recall_all@1 | 0.3333 |
| session_recall_all@10 | 1.0000 |
| session_recall_all@3 | 1.0000 |
| session_recall_all@30 | 1.0000 |
| session_recall_all@5 | 1.0000 |
| session_recall_all@50 | 1.0000 |
| session_recall_any@1 | 1.0000 |
| session_recall_any@10 | 1.0000 |
| session_recall_any@3 | 1.0000 |
| session_recall_any@30 | 1.0000 |
| session_recall_any@5 | 1.0000 |
| session_recall_any@50 | 1.0000 |
| turn_evidence_recall@1 | 0.5000 |
| turn_evidence_recall@10 | 1.0000 |
| turn_evidence_recall@3 | 0.8333 |
| turn_evidence_recall@30 | 1.0000 |
| turn_evidence_recall@5 | 1.0000 |
| turn_evidence_recall@50 | 1.0000 |
| turn_ndcg_any@1 | 0.6667 |
| turn_ndcg_any@10 | 0.7924 |
| turn_ndcg_any@3 | 0.7044 |
| turn_ndcg_any@30 | 0.7924 |
| turn_ndcg_any@5 | 0.7924 |
| turn_ndcg_any@50 | 0.7924 |
| turn_recall_all@1 | 0.3333 |
| turn_recall_all@10 | 1.0000 |
| turn_recall_all@3 | 0.6667 |
| turn_recall_all@30 | 1.0000 |
| turn_recall_all@5 | 1.0000 |
| turn_recall_all@50 | 1.0000 |
| turn_recall_any@1 | 0.6667 |
| turn_recall_any@10 | 1.0000 |
| turn_recall_any@3 | 1.0000 |
| turn_recall_any@30 | 1.0000 |
| turn_recall_any@5 | 1.0000 |
| turn_recall_any@50 | 1.0000 |

## Runtime

- Completed: 3
- Failed: 0
- Average latency: 42.927s
- Median latency: 55.908s
- P95 latency: 64.278s

## Agent trajectory

- Questions: 3
- Average steps: 6.33
- Tool counts: `{"explore_memory": 3, "read_session": 4, "search_documents": 7, "submit_evidence": 1}`
- Stop reasons: `{"max_steps": 2, "submit_evidence": 1}`

## Run configuration

- Track: `skill-agent`
- Remember profile: `strong-v1`
- Max agent steps: `8`
- Answer top-k: `10`
- Git commit: `unknown`
- Dataset SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Models/chunking: `{"chunking": {"overlap": 100, "window_size": 800}, "embedding": {"device": "cpu", "model": "all-MiniLM-L6-v2", "model_path": null}, "llm": {"agent_max_tokens": null, "agent_think": null, "agent_thinking_max_tokens": null, "answer_think": null, "answer_validation_retries": null, "api_key_env": null, "base_url": "http://sz-infer.x2robot.cn/infer/inf-dddgcurffgz3366q/v1", "context_window_tokens": 32000, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}, "max_tokens": 16384, "model": "kimi-k3", "temperature": 0.0, "think": false}, "pipeline": {"alignment": {"max_alignment_candidates": null}, "debug": {"distill_data_dir": null}, "extraction": {"entity_post_enhancement": false, "entity_refine_rounds": 0, "prompt_episode_max_chars": 2000, "relation_refine_rounds": 0}, "remember": {"abstract_recall_rounds": 1, "alignment_policy": "conservative", "anchor_recall_rounds": 1, "coverage_gap_rounds": 0, "episode_slice_chars": 800, "fallback_cooccurrence_relations": false, "family_write_gate_enabled": true, "max_entities_per_window": 24, "max_relations_per_window": 36, "mode": "multi_step", "named_entity_recall_rounds": 1, "overlap_chars": 300, "preserve_source_language": false, "profile": "strong-v1", "relation_expand_rounds": 0, "relation_hint_rounds": 1, "relation_write_rounds": 1, "window_batch_alignment": true, "window_size_chars": 6000}, "search": {"content_snippet_length": 50, "embedding_full_search_threshold": null, "embedding_name_search_threshold": null, "jaccard_search_threshold": null, "max_similar_entities": 10, "relation_content_snippet_length": 50, "relation_endpoint_embedding_threshold": 0.9, "relation_endpoint_jaccard_threshold": 0.9, "similarity_threshold": 0.7}}}`

## Failure attribution

`{"agent_stopped_early": 2}`
