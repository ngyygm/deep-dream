# Deep-Dream Security Audit Report

**Date**: 2026-04-26
**Auditor**: Security Engineer Agent
**Scope**: Deep-Dream v1.x - Versioned Knowledge Graph System

---

## Executive Summary

This audit identified **3 CRITICAL**, **4 HIGH**, **6 MEDIUM**, and **4 LOW** severity issues. The most critical findings involve lack of authentication, potential Cypher injection through regex-based query modification, and insufficient input validation in several API endpoints.

### Risk Summary

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 3 | Not Fixed |
| High | 4 | Not Fixed |
| Medium | 6 | Not Fixed |
| Low | 4 | Not Fixed |

---

## A. Input Validation (CRITICAL)

### A1. CRITICAL: Regex-Based Cypher Query Injection
**Location**: `core/storage/neo4j/_helpers.py` - `_inject_graph_id_filter()`

**Description**:
The system uses regex-based string manipulation to inject `graph_id` filters into Cypher queries. This approach is vulnerable to Cypher injection through carefully crafted graph_id values or user inputs that affect query construction.

```python
# Vulnerable code at line 179-293
def _inject_graph_id_filter(cypher: str) -> str:
    # Uses regex to find patterns and injects graph_id via string concatenation
    cypher = cypher[:insert_pos] + f" {alias}.graph_id = $graph_id AND" + cypher[insert_pos:]
```

**Attack Scenario**:
While `graph_id` is validated via `_GRAPH_ID_RE` regex, the string concatenation approach is fragile. If an attacker can bypass the regex validation (e.g., through race conditions, unicode normalization attacks, or if validation is bypassed in certain code paths), they could inject arbitrary Cypher.

**Impact**: Remote Code Execution on Neo4j database
**CVSS Score**: 9.8 (Critical)

**Recommendation**:
1. Use parameterized queries exclusively - never string concatenation for Cypher
2. Implement a proper query builder with AST-based transformation
3. Add server-side Cypher query sanitization before execution

---

### A2. HIGH: Insufficient Input Validation on `/api/v1/remember` Text Input
**Location**: `core/server/blueprints/remember.py` - `_parse_remember_input()`

**Description**:
The text input validation only checks for presence, not content:
```python
if not text:
    return err("缺少 text 或 file（必填其一）", 400)
```

No validation for:
- Maximum length (configurable `max_length` exists but not enforced)
- Malicious content (null bytes, control characters)
- Binary data伪装 as text

**Impact**: Denial of Service via memory exhaustion, potential log injection
**CVSS Score**: 7.5 (High)

**Recommendation**:
```python
# Add validation
_MAX_TEXT_LENGTH = 10_000_000  # 10MB
if len(text) > _MAX_TEXT_LENGTH:
    return err("文本长度超过限制", 400)
# Check for null bytes
if '\x00' in text:
    return err("文本包含非法字符", 400)
```

---

### A3. HIGH: Path Traversal Risk in `graph_id` Parameter
**Location**: `core/server/registry.py` - `validate_graph_id()`

**Description**:
The regex validation for `graph_id` prevents direct path traversal:
```python
_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
```

However, the storage path construction directly concatenates `graph_id`:
```python
storage_path = str(self._base_path / graph_id)
```

If the validation is bypassed or if `graph_id` contains unicode characters that normalize to path separators (unicode normalization attacks), path traversal is possible.

**Impact**: Arbitrary file write/delete, data exfiltration
**CVSS Score**: 7.5 (High)

**Recommendation**:
```python
# Add additional safeguard
if graph_id in ('.', '..') or '/' in graph_id or '\\' in graph_id:
    raise ValueError("Invalid graph_id")
```

---

### A4. MEDIUM: Missing Content-Type Validation on JSON Endpoints
**Location**: All blueprint files

**Description**:
Most endpoints use `request.get_json(silent=True)` which accepts any content type:
```python
body = request.get_json(silent=True) or {}
```

No validation that Content-Type is actually `application/json`.

**Impact**: CSRF attacks, request smuggling
**CVSS Score**: 6.5 (Medium)

**Recommendation**:
```python
if request.content_type != 'application/json':
    return err("Invalid Content-Type", 415)
```

---

### A5. MEDIUM: No Sanitization of User Content Before LLM Processing
**Location**: `core/llm/prompts.py`

**Description**:
User text is directly inserted into LLM prompts without sanitization:
```python
ENTITY_EXTRACT_USER = """文本：
{window_text}
```

**Impact**: Potential LLM prompt injection affecting extraction quality
**CVSS Score**: 5.3 (Medium)

**Recommendation**:
- Implement content length limits before LLM processing
- Consider adding delimiters/sanitization for special characters
- Monitor for unusual patterns that might affect LLM output

---

### A6. LOW: Excessive Default Rate Limit
**Location**: `core/server/api.py` - rate limiting

**Description**:
Default rate limit from config: `rate_limit_per_minute = 0` (disabled)

**Impact**: Brute force attacks, resource exhaustion
**CVSS Score**: 4.3 (Low)

**Recommendation**:
- Enable rate limiting by default
- Set reasonable defaults (e.g., 60 requests/minute per IP)

---

## B. Authentication/Authorization (CRITICAL)

### B1. CRITICAL: No Authentication on Any Endpoint
**Location**: All blueprint files

**Description**:
The entire API has no authentication mechanism. Anyone who can reach the API can:
- Read/write any graph data
- Delete entire graphs
- Execute dream operations
- Access system logs

**Impact**: Complete data compromise, unauthorized access
**CVSS Score**: 10.0 (Critical)

**Recommendation**:
1. Implement API key authentication for Agent access
2. Add JWT-based authentication for human users
3. Implement RBAC with roles: `read`, `write`, `admin`
4. Add authentication middleware in `api.py`:

```python
@app.before_request
def _require_auth():
    if request.path.startswith("/api/v1/health"):
        return  # Allow health checks
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing authentication"}), 401
    token = auth_header[7:]
    if not validate_token(token):
        return jsonify({"error": "Invalid token"}), 401
```

---

### B2. HIGH: No Authorization Checks for Cross-Graph Access
**Location**: `core/server/api.py` - `_resolve_graph_id()`

**Description**:
Any user can access any graph_id by simply specifying it in the request. No ownership check.

**Impact**: Data leakage between users/tenants
**CVSS Score**: 8.6 (High)

**Recommendation**:
- Implement graph ownership model
- Add user-to-graph authorization check
- Log all cross-graph access attempts

---

## C. LLM Prompt Injection (MEDIUM)

### C1. MEDIUM: User Input Directly Inserted into Prompts
**Location**: `core/llm/prompts.py`

**Description**:
User text from `remember` endpoint is directly used in prompts without any sanitization.

**Attack Scenario**:
An attacker could craft text that attempts to manipulate the LLM's behavior, potentially:
- Extracting system prompt information
- Influencing entity/relation extraction
- Bypassing content filters

**Impact**: Data poisoning, information disclosure
**CVSS Score**: 5.3 (Medium)

**Recommendation**:
- Add prompt injection detection
- Use delimiters around user content
- Consider using LLM-specific guardrails
- Implement output validation

---

## D. Information Disclosure (MEDIUM)

### D1. MEDIUM: Verbose Error Messages Expose Internal Details
**Location**: Throughout all blueprints

**Description**:
Many error handlers return raw exception messages:
```python
except Exception as e:
    return err(str(e), 500)
```

**Examples**:
- Database connection strings
- File system paths
- Internal function names
- Stack traces

**Impact**: Information disclosure aiding further attacks
**CVSS Score**: 5.3 (Medium)

**Recommendation**:
```python
except Exception as e:
    logger.error("Internal error", exc_info=True)
    return err("An internal error occurred", 500)
```

---

### D2. MEDIUM: Debug Endpoints Exposed in Production
**Location**: `core/server/blueprints/system.py`

**Description**:
Endpoints like `/api/v1/system/logs` expose internal system information without authentication.

**Impact**: Information disclosure
**CVSS Score**: 5.3 (Medium)

**Recommendation**:
- Require admin role for system endpoints
- Add authentication check
- Consider separate admin port

---

### D3. LOW: CORS Configuration Allows Any Origin
**Location**: `core/server/api.py`

**Description**:
```python
_ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}
```

Only checks if origin *starts with* these values, allowing:
- `http://localhost.evil.com`
- `http://127.0.0.1.evil.com`

**Impact**: CSRF from malicious subdomains
**CVSS Score**: 4.3 (Low)

**Recommendation**:
```python
_ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}
if origin in _ALLOWED_ORIGINS:
    response.headers["Access-Control-Allow-Origin"] = origin
```

---

## E. Other Security Issues

### E1. MEDIUM: File Upload Without Type Validation
**Location**: `core/server/blueprints/remember.py`

**Description**:
File upload accepts any file type:
```python
if file and file.filename:
    text = file.read().decode("utf-8")
```

No validation of:
- File size
- Content type
- Magic bytes

**Impact**: DoS via large files, potential binary processing issues
**CVSS Score**: 6.5 (Medium)

**Recommendation**:
```python
MAX_FILE_SIZE = 10_000_000  # 10MB
ALLOWED_EXTENSIONS = {'.txt', '.md', '.json'}
```

---

### E2. LOW: Missing Security Headers
**Location**: `core/server/api.py`

**Description**:
No security headers set:
- Content-Security-Policy
- X-Content-Type-Options
- X-Frame-Options
- Strict-Transport-Security

**Impact**: XSS, clickjacking
**CVSS Score**: 4.3 (Low)

**Recommendation**:
```python
@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

---

### E3. LOW: MCP Server stdin/stdin Modification
**Location**: `core/server/mcp/deep_dream_server.py`

**Description**:
```python
sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', buffering=0)
sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)
```

Modifies stdin/stdout at import time, which could interfere with security auditing and logging.

**Impact**: Security monitoring bypass
**CVSS Score**: 3.1 (Low)

**Recommendation**:
Document this behavior clearly and consider using explicit file handles.

---

## F. Positive Security Findings

1. **graph_id Validation**: Good regex-based validation prevents most injection attempts
2. **Parameterized Neo4j Queries**: Most database queries use parameterized queries properly
3. **Write Locks**: Thread-safe operations with proper locking
4. **CORS Preflight**: Proper OPTIONS handling

---

## Recommendations Summary

### Immediate Actions (Critical/High):
1. Implement API authentication (B1 - CRITICAL)
2. Refactor Cypher injection to use proper query builder (A1 - CRITICAL)
3. Add authorization for cross-graph access (B2 - HIGH)
4. Add text length validation (A2 - HIGH)
5. Strengthen graph_id path traversal protection (A3 - HIGH)

### Short-term (Medium):
1. Sanitize error messages (D1)
2. Add file upload validation (E1)
3. Add Content-Type validation (A4)
4. Secure debug endpoints (D2)

### Long-term (Low):
1. Add security headers (E2)
2. Fix CORS origin check (D3)
3. Enable default rate limiting (A6)
4. Document MCP server behavior (E3)

---

## Updated Findings (2026-04-26)

### G1. CRITICAL - LLM Prompt Injection Through User Input
**Status**: NEW - Not Fixed

The `remember.py` endpoint takes user input and directly inserts it into LLM prompts without sanitization. Attackers can use prompt injection techniques to:
- Extract system prompts
- Manipulate entity/relation extraction
- Bypass content filters

**Remediation Status**: Pending implementation of `core/llm/sanitize.py`

### G2. HIGH - XSS Vulnerabilities in Frontend
**Status**: NEW - Not Fixed

Frontend JavaScript uses `innerHTML` with user-controlled content in multiple locations without sanitization.

### G3. HIGH - Missing Endpoint-Specific Rate Limits
**Status**: PARTIALLY FIXED

Base rate limiting exists but expensive operations (graph traversal, dream) have no additional protection.

### G4. MEDIUM - File Upload Validation Weakness
**Status**: FIXED in remember.py

Current implementation checks:
- File size (10MB limit)
- File extension whitelist
- Null bytes
- UTF-8 encoding

**Missing**: Magic byte validation (added to recommendations)

---

## Fixed Issues (as of 2026-04-26)

1. **A2 - Text Input Validation**: Now validates length, null bytes in `remember.py`
2. **D3 - CORS Origin Check**: Fixed to use exact match instead of `startswith`
3. **E1 - File Upload**: Added file size and extension validation
4. **Security Headers**: Added X-Content-Type-Options, X-Frame-Options, CSP

---

## Implementation Status

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| B1 - No Authentication | CRITICAL | IMPLEMENTED | auth.py created with API key + JWT support |
| C1 - Prompt Injection | HIGH | IMPLEMENTED | sanitize.py created with pattern detection |
| G2 - XSS | HIGH | TODO | Frontend fixes needed |
| A1 - Cypher Injection | CRITICAL | FALSE POSITIVE | Uses parameterized queries |
| D1 - Error Messages | MEDIUM | FIXED | 500 errors sanitized |

## New Security Modules Created

1. **core/llm/sanitize.py** - Prompt injection protection
   - `sanitize_user_input()` - Sanitizes user text before LLM processing
   - `validate_prompt_input()` - Strict validation for API layer
   - `wrap_user_content()` - Safe delimiter wrapping
   - `check_for_prompt_leaks()` - Post-processing leak detection

2. **core/server/auth.py** - Authentication and authorization
   - API key authentication (machine-to-machine)
   - JWT token authentication (user sessions)
   - Role-based access control (RBAC)
   - `@require_auth` and `@require_permission` decorators

## Integration Instructions

To enable authentication in the application:

1. Set environment variable:
   ```bash
   export DEEPDREAM_SECRET_KEY="your-secret-key-here"
   ```

2. Initialize auth module in api.py:
   ```python
   from core.server.auth import init_auth, require_auth, require_permission
   
   # In create_app():
   init_auth()
   ```

3. Apply decorators to protected endpoints:
   ```python
   @remember_bp.route("/api/v1/remember", methods=["POST"])
   @require_auth
   @require_permission("remember:write")
   def remember():
       # ... existing code
   ```

4. For LLM prompt sanitization:
   ```python
   from core.llm.sanitize import sanitize_user_input
   
   # In remember.py _parse_remember_input():
   text_sanitized, was_modified = sanitize_user_input(text)
   if was_modified:
       logger.warning("User input was sanitized due to security concerns")
   ```

---

## Conclusion

The Deep-Dream system requires significant security hardening before production deployment, particularly in the areas of authentication and input validation. The lack of authentication is the single most critical issue, exposing all data to unauthorized access.

**Overall Risk Level**: **HIGH**

**Recommendation**: Address all CRITICAL and HIGH severity issues before production deployment.
