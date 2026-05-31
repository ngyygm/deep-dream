# Deep-Dream CLI Redesign Spec

**Date**: 2026-06-01
**Framework**: Click + Rich
**Status**: Draft

## 1. Executive Summary

Redesign `deep-dream` CLI from 1300-line monolithic argparse to modular Click+Rich architecture. All existing commands preserved with same names; adds human-readable output, shell completion, config management, server lifecycle, and safe-destructive defaults.

## 2. Architecture

### 2.1 File Structure

```
core/cli/
├── __init__.py          # Re-exports cli, main
├── _main.py             # Root Click group + global options
├── _ctx.py              # CliContext class (config, storage helpers)
├── _output.py           # Rich/JSON dual output, tables, spinner, error_out
├── _exit_codes.py       # OK=0, ERROR=1, ARGS=2, AUTH=3, NOT_FOUND=5
├── cmd_doctor.py
├── cmd_config.py
├── cmd_server.py
├── cmd_library.py
├── cmd_graph.py
├── cmd_vault.py
├── cmd_remember.py
├── cmd_find.py
├── cmd_trace.py
├── cmd_explore.py
├── cmd_docs.py
├── cmd_episode.py
├── cmd_concept.py
├── cmd_relation.py
├── cmd_sql.py
├── cmd_db.py
├── cmd_task.py
├── cmd_completion.py
└── cmd_version.py
```

### 2.2 Entry Point

```toml
# pyproject.toml (unchanged)
[project.scripts]
deep-dream = "core.cli:main"

# New dependencies
dependencies = [
    # ... existing ...
    "click>=8.1",
    "rich>=13.0",
]
```

### 2.3 Exit Codes

```python
# core/cli/_exit_codes.py
OK = 0          # Success
ERROR = 1       # General / runtime error
ARGS = 2        # Invalid arguments
AUTH = 3        # Authentication / API key issue
NOT_FOUND = 5   # Resource not found
```

### 2.4 Output Formatting

All commands output through `_output.py`:

- **Default**: Rich tables, colored panels, progress spinners (human-friendly)
- **`--json`**: Machine-readable JSON to stdout
- **`--quiet`**: Suppress non-essential output
- **`--no-color`**: Disable Rich color (also auto-disabled when piped)

### 2.5 Error Format

```
  Error: <concise description>
    Hint: <suggested fix or next step>
```

Examples:
- `Error: Graph not found: my-proj` / `Hint: Use 'deep-dream graph list' to see available graphs.`
- `Error: Only SELECT queries are allowed.` / `Hint: Use 'deep-dream db' commands for maintenance operations.`
- `Error: Provide --file or --text.` / `Hint: Example: deep-dream remember --file notes.md`
- `Error: Cannot reach server.` / `Hint: Start with 'deep-dream server start'.`

## 3. Global Options

All commands inherit these from the root group:

| Option | Short | Env Var | Default | Description |
|--------|-------|---------|---------|-------------|
| `--config` | | `DEEPDREAM_CONFIG` | `service_config.json` | Config file path |
| `--json` | | `DEEPDREAM_JSON_OUTPUT` | off | Machine-readable JSON output |
| `--no-color` | | | off | Disable colored output |
| `--quiet` | `-q` | | off | Suppress non-essential output |
| `--verbose` | `-v` | | off | Show extra diagnostic output |
| `--dry-run` | | | off | Preview without making changes |
| `--version` | | | | Show version and exit |
| `--help` | `-h` | | | Show help and exit |

Click's `auto_envvar_prefix="DEEPDREAM"` enables env var override for all options.

## 4. Command Tree

### 4.1 New Commands (not in current CLI)

| Command | Purpose |
|---------|---------|
| `version` | Show version, Python, storage path |
| `completion bash/zsh/fish` | Generate shell completion scripts |
| `config show/get/set` | Configuration management |
| `server start/stop/status` | Server lifecycle management |
| `task list/status` | Task queue management (requires server) |

### 4.2 Full Command Reference

#### `deep-dream --help`

```
Usage: deep-dream [OPTIONS] COMMAND [ARGS]...

  Deep-Dream: document-first concept graph knowledge server.

  Quick start:
    deep-dream doctor                    # Check health & config
    deep-dream remember --file notes.md  # Ingest a document
    deep-dream find "machine learning"   # Search concepts
    deep-dream explore "how does X work" # Deep exploration

  Output modes:
    --json     Machine-readable JSON output
    (default)  Human-friendly Rich tables & panels

Options:
  --version         Show version and exit
  --config PATH     Config file path [default: service_config.json]
  --json            Output raw JSON
  --no-color        Disable colored output
  -q, --quiet       Suppress non-essential output
  -v, --verbose     Show diagnostic output
  --dry-run         Preview without making changes
  -h, --help        Show this message and exit

Commands:
  completion   Generate shell completion scripts
  concept      Concept search, trace, and neighbors
  config       Configuration management
  db           Database maintenance and schema management
  docs         Document discovery and search
  doctor       Health check & diagnostics
  episode      Episode mapping helpers
  explore      Multi-strategy deep exploration
  find         Quick concept search (BM25)
  graph        Graph compatibility commands
  library      Library-level operations
  relation     Relation evidence helpers
  remember     Ingest text or file into the graph
  server       Server lifecycle management
  sql          Run read-only SQL against the graph database
  task         Task queue management (requires server)
  trace        Trace concept provenance
  vault        Vault indexing operations
  version      Show version information
```

#### `deep-dream doctor`

```
Usage: deep-dream doctor [OPTIONS]

  Inspect local Deep-Dream configuration, storage, and API health.

  Examples:
    deep-dream doctor
    deep-dream doctor --json
    deep-dream doctor --api-base http://localhost:5001/api/v1

Options:
  --api-base TEXT  API base URL [default: http://127.0.0.1:16200/api/v1]
  -h, --help       Show this message and exit.
```

**Human output:**
```
Deep-Dream Doctor

  Storage:  /home/user/deep-dream/library
  Graphs:   1
  API:      online (http://127.0.0.1:16200/api/v1)

          Graphs
┌─────────┬───────────┬──────────┬───────┐
│ ID      │ Documents │ Concepts │ Edges │
├─────────┼───────────┼──────────┼───────┤
│ library │ 42        │ 187      │ 56    │
└─────────┴───────────┴──────────┴───────┘
```

#### `deep-dream config show/get/set`

```
Usage: deep-dream config [COMMAND]...

  View and manage Deep-Dream configuration.

  Examples:
    deep-dream config show
    deep-dream config get llm.model
    deep-dream config set llm.model gpt-4o

Commands:
  show  Display resolved configuration
  get   Get a config value by dot-path
  set   Set a config value (with confirmation)

Options:
  -h, --help  Show this message and exit.
```

- API keys redacted by default; `--secrets` reveals them
- `config set` prompts for confirmation; `--yes` skips
- Dot-path keys: `llm.model`, `pipeline.search.similarity_threshold`

#### `deep-dream server start/stop/status`

```
Usage: deep-dream server [COMMAND]...

  Start, stop, and check the Deep-Dream Flask server.

  Examples:
    deep-dream server start
    deep-dream server start --port 5001 --detach
    deep-dream server status
    deep-dream server stop

Commands:
  start   Start the Flask server
  stop    Stop a running server (with confirmation)
  status  Check whether the server is running
```

- `start --detach` runs in background
- `stop` requires `--yes` to skip confirmation

#### `deep-dream graph create/list/use/stats/rebuild`

```
Usage: deep-dream graph [COMMAND]...

  Graph management commands.

  Examples:
    deep-dream graph list
    deep-dream graph create my-project
    deep-dream graph stats
    deep-dream graph rebuild --dry-run

Commands:
  list     List all graphs
  create   Create a new graph
  use      Set the active graph
  stats    Show graph statistics
  rebuild  Clear graph data for re-indexing (DANGEROUS)
```

- `rebuild` requires `--yes` confirmation or `--dry-run` preview

#### `deep-dream remember`

```
Usage: deep-dream remember [OPTIONS]

  Run the remember pipeline on a file or inline text.
  Provide exactly one of --file or --text.

  Examples:
    deep-dream remember --file notes.md
    deep-dream remember --text "Key insight about quantum computing"
    deep-dream remember --file doc.md --source "research-paper" -v

Options:
  --file PATH     File to remember
  --text TEXT     Inline text to remember
  --source TEXT   Source label [default: file name or "cli:text"]
  --encoding TEXT File encoding [default: utf-8]
  --graph TEXT    Graph ID [default: active]
  -v, --verbose   Show processing details
  -h, --help      Show this message and exit.
```

- Shows Rich progress spinner during processing
- `--verbose` outputs pipeline stage details

#### `deep-dream find`

```
Usage: deep-dream find [OPTIONS] QUERY

  Search concepts by keyword (BM25 full-text search).
  For semantic search, use 'deep-dream concept search --semantic'.

  Examples:
    deep-dream find "machine learning"
    deep-dream find "transformer" --role entity --limit 10

Options:
  --graph TEXT    Graph ID [default: active]
  --role TEXT     Filter: document|episode|entity|relation
  --limit INT     Max results [default: 20]
  --time-point TEXT  Temporal snapshot ISO timestamp
  -h, --help      Show this message and exit.
```

#### `deep-dream explore`

```
Usage: deep-dream explore [OPTIONS] QUESTION

  Document-first semantic and graph exploration.
  Combines file search, concept search, graph traversal, and relation evidence.

  Examples:
    deep-dream explore "how does attention mechanism work"
    deep-dream explore "RAG pipeline design" --limit 30
    deep-dream explore "causal inference" --terms "do-calculus,counterfactual,dag"
```

#### `deep-dream docs roots/list/path/search/grep/map`

```
Usage: deep-dream docs [COMMAND]...

  Document discovery, search, and mapping.

  Examples:
    deep-dream docs list --limit 20
    deep-dream docs search "attention mechanism"
    deep-dream docs grep "transformer.*layer"
    deep-dream docs map ./notes/ml-paper.md

Commands:
  roots   List searchable document roots
  list    List indexed documents
  path    Resolve document ID to file path
  search  Literal text search
  grep    Regex text search
  map     Map file path to document records
```

#### `deep-dream db init-v15/reset-v15/rebuild-fts/validate/rebuild-current/vacuum-embeddings/compact`

```
Usage: deep-dream db [COMMAND]...

  Database maintenance, schema management, and integrity tools.

  Examples:
    deep-dream db validate
    deep-dream db validate --repair
    deep-dream db rebuild-fts
    deep-dream db compact
    deep-dream db vacuum-embeddings --dry-run

Commands:
  init-v15           Initialize V1.5 schema
  reset-v15          Backup and reset database (DANGEROUS)
  rebuild-fts        Rebuild full-text search index
  validate           Run integrity validation
  rebuild-current    Rebuild content/current/ files
  vacuum-embeddings  Clean orphaned embeddings
  compact            VACUUM to reclaim disk space
```

- `reset-v15` requires `--backup-old` AND confirmation
- `vacuum-embeddings` supports `--dry-run`
- `compact` requires `--yes` confirmation

## 5. Design Principles Compliance

| # | Principle | Status | How Applied |
|---|-----------|--------|-------------|
| 1 | Task-first design | SATISFIED | High-freq: `find`, `remember`, `doctor`. Low-freq: `db vacuum-embeddings` |
| 2 | Natural language | SATISFIED | Consistent `<resource> <action>` pattern throughout |
| 3 | Safe defaults | SATISFIED | Destructive ops require `--yes`; `--dry-run` on `rebuild`, `migrate`, `vacuum` |
| 4 | Consistent params | SATISFIED | `--graph`, `--limit`, `--force` used uniformly; Click enforces types |
| 5 | Useful --help | SATISFIED | Every command has examples, defaults, descriptions in docstring |
| 6 | Dual output | SATISFIED | Rich tables by default; `--json` for machines; same data shape |
| 7 | stdout/stderr/exit | SATISFIED | stdout for results, stderr for Rich output, semantic exit codes |
| 8 | Actionable errors | SATISFIED | `Error:` + `Hint:` pattern with suggested commands |
| 9 | Automation | SATISFIED | `--yes`, `--json`, `--quiet`, `--no-color`; env vars via Click |
| 10 | Config transparency | SATISFIED | `config show/get/set` with dot-path, JSON parsing, key redaction |
| 11 | Controlled interactivity | SATISFIED | Click detects TTY; `--yes` for scripts; Rich auto-disables in pipe |
| 12 | Progress | SATISFIED | Rich spinner on `remember`, `vault index`, `explore` |
| 13 | Undo/preview | SATISFIED | `--dry-run` on destructive ops; backup before `reset-v15` |
| 14 | Performance | PARTIAL | Lazy imports in Click; `--help`/`--version` fast via Click lazy groups |
| 15 | Shell completion | SATISFIED | `completion bash/zsh/fish`; Click generates completion scripts |
| 16 | Backward compat | SATISFIED | All existing command names preserved; deprecated commands show warning |

## 6. Verification Notes

The verify agent found these items to watch during implementation:

1. **Duplicate commands**: `find` overlaps `concept search` (BM25); `trace` overlaps `concept trace`. Keep both for ergonomics but note the overlap in help text.
2. **`--graph` flag**: Under single-library model, this is vestigial. Keep for backward compat but mark as optional/default in help.
3. **Lazy imports**: `_main.py` must use lazy imports to avoid loading the entire pipeline on `--help`. Use Click's `lazy=True` or deferred imports.
4. **Helper functions**: Many commands delegate to helper functions from the legacy `core/cli.py` (e.g. `_document_rows`, `_search_document_files`, `_resolve_concept_id`). These should be moved to a shared module or kept in legacy file and imported.
5. **Server commands**: `server start` and `task list/status` require the server to be running. Error messages should suggest `deep-dream server start`.

## 7. Implementation Strategy

1. Create `core/cli/` package with `_main.py`, `_ctx.py`, `_output.py`, `_exit_codes.py`
2. Implement `cmd_version.py`, `cmd_doctor.py`, `cmd_config.py` first (new commands)
3. Migrate each existing command one at a time from legacy `core/cli.py`
4. Keep legacy `core/cli.py` as `core/cli_legacy.py` for backward compat during transition
5. Update `pyproject.toml` entry point and dependencies
6. Add tests per command
