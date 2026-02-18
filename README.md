# Power Interpreter MCP

> A production-grade sandboxed Python execution engine with persistent sessions, inline chart rendering, and file delivery — built as an MCP server for SimTheory.ai.

[![Deployed on Railway](https://img.shields.io/badge/Deployed-Railway-blueviolet)](https://railway.app)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP%201.6-green)](https://modelcontextprotocol.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

---

## What It Does

Power Interpreter gives AI agents (via MCP) the ability to:

- **Execute Python code** in a secure sandbox with 30+ pre-installed libraries
- **Persist session state** — variables, DataFrames, and imports survive across calls (like Jupyter)
- **Generate charts** that render inline in the chat (matplotlib, seaborn, plotly)
- **Create downloadable files** (Excel, PDF, CSV) with click-to-download URLs
- **Load and query large datasets** via PostgreSQL (500K+ rows, survives restarts)
- **Run long jobs asynchronously** with polling (up to 30-minute timeout)

### vs. OpenAI Code Interpreter

| Capability | Code Interpreter | Power Interpreter |
|---|---|---|
| Dataset size | ~100MB, crashes on large files | 512K+ rows, 57MB loaded in 69s via chunked PostgreSQL |
| Data persistence | Dies with session | PostgreSQL — survives restarts, queryable anytime |
| Long-running jobs | 60-second hard timeout | `submit_job` with 30-min timeout, background execution |
| External data | Upload through chat UI only | `fetch_file` pulls from Google Drive, URLs, APIs |
| Infrastructure | Black box | Full Railway logs, real-time debugging |
| Integration | OpenAI only | MCP protocol — SimTheory, Claude Desktop, any MCP client |
| Session state | ✅ Persistent | ✅ Persistent (KernelManager with idle timeout) |
| Inline charts | ✅ Automatic | ✅ Automatic (matplotlib auto-capture + Postgres storage) |
| File downloads | ✅ Automatic | ✅ Automatic (Postgres-backed `/dl/` URLs) |
| Domain-specific tools | ❌ Never | 🔜 Custom financial analysis tools (roadmap) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SimTheory.ai / Claude Desktop / Any MCP Client                │
│  (sends JSON-RPC over SSE)                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP Protocol (JSON-RPC 2.0)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Application (main.py)                                  │
│  ├── /mcp/sse          → MCP SSE endpoint (mcp_server.py)      │
│  ├── /api/execute      → Sync code execution                   │
│  ├── /api/jobs/*       → Async job queue                       │
│  ├── /api/data/*       → Dataset load/query                    │
│  ├── /api/files/*      → File upload/download/list             │
│  ├── /api/sessions/*   → Session management                    │
│  ├── /dl/{id}/{name}   → Public file download (Postgres-backed)│
│  └── /health           → Health check                          │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────────────┐
│  Sandbox Executor   │          │  PostgreSQL                    │
│  (executor.py)      │          │  ├── datasets (loaded CSVs)    │
│  ├── KernelManager  │          │  ├── sandbox_files (binary)    │
│  │   (persistent    │          │  ├── jobs (async queue)        │
│  │    session state) │          │  └── sessions (metadata)      │
│  ├── ChartCapture   │          └────────────────────────────────┘
│  │   (auto-capture  │
│  │    matplotlib)    │
│  ├── Import Allow-  │
│  │   list (_lazy_   │
│  │   import)        │
│  └── Safe file I/O  │
│      (sandboxed)    │
└─────────────────────┘
```

### Key Data Flows

**Code Execution with Charts:**
```
MCP tools/call 'execute_code' {code, session_id}
  → executor.execute(code, session_id)
    → KernelManager: get_or_create persistent globals
    → _preprocess_code: rewrite imports via allowlist
    → exec(code, sandbox_globals)  [state persists!]
    → ChartCapture: auto-save unclosed matplotlib figures as PNG
    → _store_files_in_postgres: PNG/Excel/PDF → Postgres with /dl/ URL
    → stdout includes inline image markdown + download links
  → MCP response: content blocks with text (URLs pass through unmodified)
  → SimTheory renders charts inline + download links clickable
```

**File Download:**
```
User clicks link → GET /dl/{file_id}/{filename}
  → Postgres lookup by file_id
  → Stream binary content with correct Content-Type
  → Browser downloads file
```

---

## MCP Tools (11 total)

| Tool | Description | Key Parameters |
|---|---|---|
| `execute_code` | Run Python code with persistent session state | `code`, `session_id` |
| `submit_job` | Submit long-running code (up to 30 min) | `code`, `session_id`, `timeout` |
| `get_job_status` | Check async job progress | `job_id` |
| `get_job_result` | Get completed job output | `job_id` |
| `upload_file` | Upload file to session sandbox | `filename`, `content` (base64) |
| `fetch_file` | Download file from URL into sandbox | `url`, `filename` |
| `list_files` | List files in session sandbox | `session_id` |
| `load_dataset` | Load CSV/Excel into PostgreSQL | `file_path`, `dataset_name` |
| `query_dataset` | Run SQL against loaded datasets | `sql`, `limit` |
| `list_datasets` | List all loaded datasets | — |
| `create_session` | Create a named workspace | `session_id` |

---

## Persistent Session State (KernelManager)

Each `session_id` gets a persistent Python namespace. Variables, imports, and DataFrames survive across `execute_code` calls — just like a Jupyter notebook.

```
Call 1: execute_code("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]})")
Call 2: execute_code("print(df.describe())")  # ← df still exists!
Call 3: execute_code("df.to_excel('output.xlsx')")  # ← still exists!
```

**How it works:**
- `KernelManager` stores `sandbox_globals` dicts per `session_id`
- First call: "Slow path" — builds fresh globals with safe builtins
- Subsequent calls: "Fast path" — reuses existing globals dict
- Idle sessions are cleaned up after timeout

---

## Inline Charts & File Downloads

### Charts (matplotlib, seaborn)
Charts are automatically captured and rendered inline in the chat:

1. **`plt.show()` interception** — captures all open figures as PNG
2. **`Figure.savefig()` tracking** — tracks explicitly saved images
3. **Post-execution sweep** — captures any unclosed figures (safety net)
4. **Postgres storage** — PNG stored with `/dl/` URL
5. **Inline markdown** — `![chart](url)` appended to stdout
6. **SimTheory renders** — chart appears inline in the conversation

### Files (Excel, PDF, CSV, etc.)
Generated files are automatically stored and delivered:

1. **File detection** — new files in session directory detected post-execution
2. **Postgres storage** — binary content stored with UUID-based `/dl/` URL
3. **Download links** — markdown links appended to stdout
4. **Click to download** — user clicks link, browser downloads file

**Supported file types:** `.xlsx`, `.xls`, `.csv`, `.tsv`, `.json`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.svg`, `.html`, `.txt`, `.md`, `.zip`, `.parquet`

---

## Pre-installed Libraries

### Data Analysis
| Library | Version | Notes |
|---|---|---|
| pandas | 2.2.3 | DataFrames, CSV/Excel I/O |
| numpy | 2.2.1 | Numerical computing |
| openpyxl | 3.1.5 | Excel read/write (.xlsx) |
| xlsxwriter | 3.2.0 | Excel write with formatting |
| pdfplumber | 0.11.4 | PDF text/table extraction |
| tabulate | 0.9.0 | Pretty-print tables |

### PDF Generation
| Library | Version | Notes |
|---|---|---|
| reportlab | 4.1.0 | Professional PDF creation with tables, styles, headers |

### Visualization
| Library | Version | Notes |
|---|---|---|
| matplotlib | 3.10.0 | Charts (auto-captured as PNG) |
| seaborn | 0.13.2 | Statistical visualizations |
| plotly | 5.24.1 | Interactive charts |

### Statistics & ML
| Library | Version | Notes |
|---|---|---|
| scipy | 1.15.1 | Scientific computing |
| scikit-learn | 1.6.1 | Machine learning |
| statsmodels | 0.14.4 | Statistical models |

### Standard Library (available in sandbox)
`math`, `statistics`, `datetime`, `collections`, `itertools`, `functools`, `re`, `json`, `csv`, `io`, `pathlib`, `copy`, `hashlib`, `base64`, `decimal`, `fractions`, `random`, `time`, `calendar`, `pprint`, `dataclasses`, `typing`, `os`, `string`, `struct`, `textwrap`

---

## Sandbox Security

The executor runs code in a controlled environment:

| Control | Implementation |
|---|---|
| **Import allowlist** | `_lazy_import()` — only whitelisted modules load; all others blocked with `# [sandbox] BLOCKED` |
| **File I/O sandboxing** | `safe_open()` — all file access restricted to session directory |
| **Blocked builtins** | `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `exit`, `quit`, `breakpoint`, `input` |
| **Resource limits** | Configurable time timeout, memory limit via `resource.setrlimit` |
| **SQL injection prevention** | Dataset queries restricted to `SELECT` only |
| **API authentication** | `X-API-Key` header required on all endpoints |

---

## Deployment

### Railway (Production)

1. Create new project in Railway
2. Connect this GitHub repo
3. Add PostgreSQL plugin
4. Set environment variables (see below)
5. Deploy — Railway auto-builds from `Dockerfile`

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_KEY` | API authentication key | (required) |
| `DATABASE_URL` | PostgreSQL connection string | (auto from Railway) |
| `PUBLIC_BASE_URL` | Public URL for file download links | (auto-detected) |
| `MAX_EXECUTION_TIME` | Max sync execution (seconds) | 300 |
| `MAX_MEMORY_MB` | Memory limit per execution | 4096 |
| `MAX_FILE_SIZE_MB` | Max upload file size | 500 |
| `MAX_CONCURRENT_JOBS` | Parallel job limit | 4 |
| `JOB_TIMEOUT` | Max async job time (seconds) | 600 |
| `SANDBOX_FILE_MAX_MB` | Max file size for Postgres storage | 50 |
| `SANDBOX_FILE_TTL_HOURS` | File expiration (0 = never) | 72 |
| `MAX_OUTPUT_SIZE` | Max stdout/stderr capture (bytes) | 100000 |

---

## API Endpoints

### Code Execution
```bash
# Sync execution
POST /api/execute
{"code": "import pandas as pd; print(pd.__version__)", "session_id": "default", "timeout": 30}

# Async job submission
POST /api/jobs/submit
{"code": "...", "session_id": "default", "timeout": 600}

# Job status
GET /api/jobs/{job_id}/status

# Job result
GET /api/jobs/{job_id}/result
```

### Data Management
```bash
# Load CSV into PostgreSQL
POST /api/data/load-csv
{"file_path": "data.csv", "dataset_name": "my_data"}

# Query dataset
POST /api/data/query
{"sql": "SELECT * FROM data_xxx WHERE amount > 100", "limit": 1000}

# List datasets
GET /api/data/datasets
```

### File Management
```bash
# Upload file
POST /api/files/upload

# List files
GET /api/files/list?session_id=default

# Download generated file (public, no auth)
GET /dl/{file_id}/{filename}
```

### MCP
```bash
# MCP SSE endpoint (used by SimTheory/Claude Desktop)
POST /mcp/sse
```

### Health
```bash
GET /health
```

---

## Project Structure

```
power-interpreter/
├── app/
│   ├── main.py              # FastAPI app + MCP SSE mount
│   ├── mcp_server.py        # MCP tool definitions (11 tools)
│   ├── config.py            # Settings from environment
│   ├── database.py          # SQLAlchemy async engine
│   ├── models.py            # DB models (SandboxFile, etc.)
│   ├── engine/
│   │   ├── executor.py      # Sandbox executor (v2.7.0)
│   │   └── kernel_manager.py # Persistent session state
│   └── routes/
│       ├── execute.py       # /api/execute
│       ├── data.py          # /api/data/*
│       ├── files.py         # /api/files/* + /dl/*
│       ├── sessions.py      # /api/sessions/*
│       └── health.py        # /health
├── requirements.txt
├── Dockerfile
├── railway.toml
└── README.md
```

---

## Version History

| Version | Date | Changes |
|---|---|---|
| **2.7.0** | 2026-02-17 | Add reportlab for PDF generation; add matplotlib PDF backend to allowlist |
| **2.6.0** | 2026-02-17 | Fix critical import alias bug (matplotlib.pyplot as plt); robust chart capture with 3-mechanism approach |
| **2.5.x** | 2026-02-17 | Fix URL stripping in mcp_server.py; inline charts + file downloads working end-to-end |
| **2.1.0** | 2026-02-17 | Auto file storage in Postgres with `/dl/` download URLs |
| **2.0.0** | 2026-02-17 | Persistent kernel sessions via KernelManager |
| **1.x** | 2026-02 | Initial release — stateless execution, dataset support, async jobs |

---

## Roadmap

| Priority | Feature | Status |
|---|---|---|
| ~~P1~~ | ~~File downloads & inline charts~~ | ✅ **Done** |
| ~~P2~~ | ~~Persistent Python kernel (session state)~~ | ✅ **Done** |
| P3 | Structured error handling + auto-retry suggestions | 🟡 Partial |
| P4 | Automatic file handling from chat uploads | 🟡 Partial |
| **P5** | **Domain-specific financial analysis tools** | ❌ Not started |

P5 is the competitive moat — custom MCP tools like `analyze_intercompany`, `reconciliation_report`, and `variance_analysis` that know your chart of accounts, entity structure, and elimination logic. Code Interpreter can never do this.

---

## Author

Built by **Kaffer AI** for **Timothy Escamilla**

## License

Private — All rights reserved
