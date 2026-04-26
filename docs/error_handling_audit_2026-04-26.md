# Error Handling Audit - 2026-04-26

## Summary

Conducted comprehensive audit of error handling and retry logic across the Deep-Dream codebase.

## Files Audited

1. **core/llm/client.py** (1380 lines) - LLM client with error handling
2. **core/llm/errors.py** - Error type definitions
3. **core/server/api.py** - API endpoints and request handling
4. **core/server/task_queue.py** (1497 lines) - Task retry logic
5. **core/storage/neo4j/_entities.py** - Neo4j entity operations
6. **core/storage/neo4j/_base.py** - Neo4j base operations and connection management
7. **core/remember/orchestrator.py** - Pipeline error handling and workers

## Findings

### 1. LLM Client (core/llm/client.py) ✅ Already Well-Implemented

**Status**: No changes needed - comprehensive retry logic already in place.

**Existing retry logic**:
- Connection errors: 3^n backoff, max 5 rounds (lines 888-903)
- Timeout errors: 3^n backoff, max 5 rounds (lines 906-923)
- Rate limit (429/TPM): Unlimited retries with exponential backoff capped at 3600s (lines 872-885)
- UTF-8 encoding errors: Up to 5 retries (lines 812-820)
- max_tokens overflow: Automatic reduction with retry (lines 861-869)

**Error classification**:
- `_is_rate_limit_tpm_error()` - Detects 429/rate limit errors
- `_CONNECTION_ERROR_KEYWORDS` - frozenset of transient connection error patterns
- `_RATE_LIMIT_KEYWORDS` - frozenset of rate limit error patterns

### 2. Neo4j Base (core/storage/neo4j/_base.py) ⚠️ Improved

**Status**: Added retry wrapper for transient connection errors.

**Changes made**:
- Added `_NEO4J_TRANSIENT_ERRORS` frozenset for error detection
- Added `_is_transient_neo4j_error()` helper function
- Added `_run_with_retry()` method with exponential backoff (max 3 retries, max delay 30s)

**Usage example**:
```python
# Before: self._run(session, cypher, **kwargs)
# After:  self._run_with_retry(session, cypher, **kwargs)
```

### 3. API Server (core/server/api.py) ⚠️ Improved

**Status**: Added input validation helpers.

**Changes made**:
- `_validate_graph_id()` - Validate graph_id parameter
- `_validate_text_input()` - Validate text fields with min/max length
- `_validate_positive_int()` - Validate integer parameters
- `_validate_float_range()` - Validate float parameters in range
- `_make_validation_error()` - Standardized error response

**Recommendation**: Blueprint modules should use these helpers for consistent validation.

### 4. Task Queue (core/server/task_queue.py) ✅ Already Well-Implemented

**Status**: No changes needed - robust retry logic in place.

**Existing features**:
- Worker retry with configurable `max_retries` and `retry_delay_seconds`
- Journal persistence for crash recovery
- Graceful handling of missing original files
- Task state management (queued/running/completed/failed/paused/cancelled)

### 5. Remember Orchestrator (core/remember/orchestrator.py) ⚠️ Improved

**Status**: Minor improvement to executor shutdown.

**Changes made**:
- Improved `prefetch_executor` shutdown with wait=True, timeout=5 before fallback to wait=False
- Ensures clean shutdown even if tasks are still running

**Existing good practices**:
- `_record_window_error()` properly records failures
- `_signal_control_stop()` broadcasts control signals
- Step6/step7 workers handle upstream failures gracefully
- Finally blocks ensure proper cleanup

## Recommendations

1. **Neo4j operations**: Consider using `_run_with_retry()` for critical operations in `_entities.py`
2. **API validation**: Apply the new validation helpers consistently across all blueprints
3. **Monitoring**: Add metrics for retry counts to identify systemic issues
4. **Documentation**: Document which operations use retry logic and their backoff strategies

## Test Coverage

Areas that could benefit from error injection testing:
- Neo4j connection failures during batch operations
- LLM API rate limiting during concurrent requests
- Task queue recovery after crash
- Prefetch executor shutdown during active workloads

## Conclusion

The codebase has solid error handling foundations. The main improvements made were:
1. Adding transient error detection and retry logic for Neo4j operations
2. Standardizing input validation helpers for API endpoints
3. Ensuring clean executor shutdown

All changes are backward compatible and follow existing code patterns.
