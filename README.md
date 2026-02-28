# Power Interpreter MCP

**General-purpose sandboxed Python execution engine with MCP integration.**

Built for [SimTheory.ai](https://simtheory.ai) — execute Python code, load datasets, generate charts, and run long-running analysis jobs, all through the Model Context Protocol (MCP).

---

## Version

**v2.8.4** — Unified version: datetime convenience aliases, universal data loading, updated MCP tool descriptions

---

## Architecture

```
SimTheory.ai (MCP Client)
    │
    ▼  JSON-RPC over HTTP POST
┌─────────────────────────────────────────┐
│  Power Interpreter (Railway)            │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ MCP Server  │  │ FastAPI Routes   │  │
│  │ (12 tools)  │──│ /api/execute     │  │
│  │             │  │ /api/data/load   │  │
│  │             │  │ /api/files/*     │  │
│  │             │  │ /api/jobs/*      │  │
│  └─────────────┘  └──────────────────┘  │
│         │                  │            │
│         ▼                  ▼            │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Python      │  │ PostgreSQL       │  │
│  │ Kernel      │  │ (datasets,       │  │
│  │ (persistent │  │  files, jobs,    │  │
│  │  sessions)  │  │  metadata)       │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
```

---

## MCP Tools (12)

### Code Execution
| Tool | Description |
|------|-------------|
| `execute_code` | Run Python code in a persistent sandbox kernel (sync, <60s) |
| `submit_job` | Submit long-running code for async execution (up to 30 min) |
| `get_job_status` | Check async job progress |
| `get_job_result` | Retrieve completed job output |

### File Management
| Tool | Description |
|------|-------------|
| `fetch_from_url` | ★ Download file from any HTTPS URL into sandbox (CDN, S3, etc.) |
| `upload_file` | Upload a file via base64 encoding (<10MB) |
| `fetch_file` | Download file from URL (legacy, use `fetch_from_url`) |
| `list_files` | List files in the sandbox |

### Data & Datasets
| Tool | Description |
|------|-------------|
| `load_dataset` | Load data file into PostgreSQL — **auto-detects format** (see below) |
| `query_dataset` | Execute SQL SELECT queries against loaded datasets |
| `list_datasets` | List all datasets in PostgreSQL |

### Sessions
| Tool | Description |
|------|-------------|
| `create_session` | Create isolated workspace session |

---

## Supported Data Formats

The `load_dataset` tool (and the `/api/data/load` endpoint) **auto-detects file format** from the file extension:

| Format | Extensions | Reader | Notes |
|--------|-----------|--------|-------|
| **CSV** | `.csv`, `.tsv`, `.txt` | `pd.read_csv()` | Chunked loading for large files |
| **Excel** | `.xlsx`, `.xls`, `.xlsm`, `.xlsb` | `pd.read_excel()` | Full read, then chunked insert |
| **PDF** | `.pdf` | `pdfplumber` | Extracts tabular data from PDF pages |
| **JSON** | `.json` | `pd.read_json()` + `json_normalize` | Array of objects or nested JSON |
| **Parquet** | `.parquet`, `.pq` | `pd.read_parquet()` | Columnar format, very fast |

All formats are loaded into PostgreSQL in 50K-row chunks with automatic indexing on date and ID columns. Handles **1.5M+ rows** efficiently.

### Typical Workflow

```
1. fetch_from_url(url="https://cdn.example.com/invoices.xlsx", filename="invoices.xlsx")
2. load_dataset(file_path="invoices.xlsx", dataset_name="invoices")
3. query_dataset(sql="SELECT vendor, SUM(amount) FROM data_xxx GROUP BY vendor")
```

---

## Sandbox Features

### Pre-loaded Globals (available without imports)
| Name | Type | Since |
|------|------|-------|
| `pd`, `pandas` | pandas module | v1.0 |
| `np`, `numpy` | numpy module | v1.0 |
| `datetime` | datetime module | v1.0 |
| `timedelta` | datetime.timedelta | **v2.8.4** |
| `timezone` | datetime.timezone | **v2.8.4** |
| `date` | datetime.date | **v2.8.4** |
| `json`, `csv`, `math`, `re`, `io`, `copy` | stdlib modules | v1.0 |
| `collections`, `itertools`, `functools` | stdlib modules | v1.0 |
| `statistics`, `hashlib`, `base64` | stdlib modules | v1.0 |
| `Decimal`, `Fraction`, `Path` | stdlib classes | v1.0 |

### Lazy-loaded Libraries (loaded on first `import`)
| Library | Aliases Set | Notes |
|---------|------------|-------|
| matplotlib | `plt` | Agg backend auto-configured, PDF backend included |
| seaborn | `sns` | |
| plotly | `px`, `go` | express + graph_objects |
| scipy | | stats, optimize, interpolate sub-modules |
| sklearn | | scikit-learn |
| statsmodels | `sm` | |
| openpyxl | | styles, utils, chart, formatting sub-modules |
| xlsxwriter | | |
| pdfplumber | | |
| reportlab | | platypus, pdfgen sub-modules (v2.7.0) |
| requests | | HTTP client |
| tabulate, textwrap, string, struct | | |
| decimal, fractions, random, time, calendar | | |
| pprint, dataclasses, typing, pathlib, os | | |
| urllib, shutil, glob | | |

### Path Safety (v2.8.0–v2.8.3)
- **Session prefix stripping**: `default/file.csv` → `file.csv` when cwd is already `/sandbox/sessions/default/`
- **/tmp/ interception**: `/tmp/output.csv` → `output.csv` in sandbox
- **Read-only upload access**: Files at `/home/ubuntu/uploads/` readable but not writable
- **Sandbox path recognition**: `/app/sandbox_data/` paths passed through unchanged

---

## API Endpoints

### Public (no auth)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/dl/{file_id}/{filename}` | Download generated files |
| `GET` | `/charts/{session_id}/{filename}` | Serve chart images |

### Protected (API key required)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/execute` | Execute Python code |
| `POST` | `/api/data/load` | Load data file (universal format) |
| `POST` | `/api/data/load-csv` | Legacy alias (now auto-detects all formats) |
| `POST` | `/api/data/query` | SQL query against datasets |
| `GET` | `/api/data/datasets` | List datasets |
| `GET` | `/api/data/datasets/{name}` | Dataset info |
| `DELETE` | `/api/data/datasets/{name}` | Drop dataset |
| `POST` | `/api/files/upload` | Upload file (base64) |
| `POST` | `/api/files/fetch` | Fetch file from URL |
| `GET` | `/api/files` | List sandbox files |
| `POST` | `/api/jobs/submit` | Submit async job |
| `GET` | `/api/jobs/{id}/status` | Job status |
| `GET` | `/api/jobs/{id}/result` | Job result |
| `POST` | `/api/sessions` | Create session |

### MCP Transport
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mcp/sse` | Direct JSON-RPC handler (SimTheory) |
| `GET` | `/mcp/sse` | SSE transport (standard MCP clients) |

---

## Deployment

Deployed on **Railway** with:
- Python 3.11+
- PostgreSQL (datasets, files, jobs, metadata)
- Uvicorn ASGI server on port 8080

### Environment Variables
| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `API_KEY` | API key for protected endpoints |
| `RAILWAY_PUBLIC_DOMAIN` | Auto-set by Railway for public URLs |

### Configuration
| Setting | Default |
|---------|---------|
| Max execution time | 300s |
| Max memory | 4096 MB |
| Max concurrent jobs | 4 |
| Job timeout | 1800s (30 min) |
| Sandbox file max size | 50 MB |
| Sandbox file TTL | 72 hours |
| Dataset chunk size | 50,000 rows |

---

## Project Structure

```
power-interpreter/
├── app/
│   ├── main.py              # FastAPI app, lifespan, MCP JSON-RPC handler (v2.8.4)
│   ├── mcp_server.py         # MCP tool definitions — 12 tools (v1.8.2)
│   ├── config.py              # Settings and environment config
│   ├── auth.py                # API key authentication
│   ├── database.py            # PostgreSQL connection management
│   ├── models.py              # SQLAlchemy models (Dataset, SandboxFile, etc.)
│   ├── engine/
│   │   ├── data_manager.py    # ★ Universal data loading (CSV/Excel/PDF/JSON/Parquet)
│   │   ├── executor.py        # ★ Sandboxed Python execution (v2.8.4)
│   │   ├── file_manager.py    # Sandbox file management
│   │   ├── job_manager.py     # Async job queue
│   │   └── kernel_manager.py  # Persistent Python kernel sessions
│   └── routes/
│       ├── data.py            # /api/data/* endpoints
│       ├── execute.py         # /api/execute endpoint
│       ├── files.py           # /api/files/* + /dl/* endpoints
│       ├── health.py          # /health endpoint
│       ├── jobs.py            # /api/jobs/* endpoints
│       └── sessions.py        # /api/sessions endpoint
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Version History

| Version | Date | Component | Changes |
|---------|------|-----------|---------|
| **v2.8.4** | 2026-02-28 | executor, main | datetime convenience aliases (timedelta, timezone, date); unified version across all components |
| **v2.8.3** | 2026-02-22 | executor | /app/sandbox_data added to allowed read paths |
| **v2.8.2** | 2026-02-22 | executor | Read-only upload access for files outside sandbox |
| **v2.8.1** | 2026-02-22 | executor | /tmp/ path interception and redirect to sandbox |
| **v2.8.0** | 2026-02-22 | executor | Defensive path normalization (doubled session prefix) |
| **v2.7.0** | 2026-02-21 | executor | reportlab + matplotlib PDF backend allowlisted |
| **v2.6** | 2026-02-20 | executor | Critical fix: matplotlib.pyplot alias override bug |
| **v1.8.2** | 2026-02-23 | mcp_server | load_dataset description updated for universal format |
| **v1.8.1** | 2026-02-23 | main, mcp_server | Chart serve route + base64 stdout regex fallback |
| **v1.8.0** | 2026-02-22 | mcp_server | Base64 ImageContent blocks for charts |
| **v1.7.2** | 2026-02-22 | mcp_server | fetch_from_url route fix (404 → correct path) |
| **v1.7.1** | 2026-02-22 | mcp_server | FastMCP constructor fix (removed unsupported kwarg) |
| **v1.7.0** | 2026-02-21 | mcp_server | fetch_from_url tool added |
| **v1.6.0** | 2026-02-20 | mcp_server | Auto file handling — tool descriptions rewritten |
| **v1.5.2** | 2026-02-19 | mcp_server | Stop stripping stdout — pass URLs through as-is |
| **v1.5.1** | 2026-02-19 | mcp_server | Plain text URL format (still broken) |
| **v1.5.0** | 2026-02-18 | mcp_server | Content blocks introduced (broke URL passing) |
| **v1.2.0** | 2026-02-15 | all | Initial working version |

---

## Smoke Test Results (v2.8.4 — 2026-02-28)

All capabilities verified:

| Category | Tests | Status |
|----------|-------|--------|
| Environment | Python 3.11+, 15/15 libraries | ✅ |
| datetime aliases | timedelta, timezone, date at top level | ✅ |
| time module safety | `import time` not shadowed | ✅ |
| Session persistence | Variables survive across calls | ✅ |
| Data analysis | Revenue, margins, pivots | ✅ |
| Chart generation | 4-panel matplotlib PNG | ✅ |
| File export | CSV, Excel (4 sheets), JSON | ✅ |
| Universal data loading | CSV, Excel, PDF, JSON, Parquet | ✅ |

---

## Author

Built by **Kaffer AI** for **Timothy Escamilla**, CEO at New Carrot Farms LLC.

Part of the AI infrastructure stack for business analytics, M&A due diligence, and operational intelligence.
