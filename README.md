# Power Interpreter MCP

**General-purpose sandboxed Python execution engine with MCP integration.**

Built for [SimTheory.ai](https://simtheory.ai) — execute Python code, load datasets, generate charts, and run long-running analysis jobs, all through the Model Context Protocol (MCP).

---

## Version

**v2.9.1** — Centralized version constant, token-optimized MCP descriptions, smart empty-argument handling, startup consistency across all components.

---

## Positioning

Power Interpreter is optimized first for **personal and practical AI workflows**:

- Python / code execution in a persistent sandbox
- Spreadsheet and dataset analysis (CSV, Excel, JSON, Parquet, PDF)
- Chart generation (matplotlib, seaborn, plotly)
- File export and persistent download URLs
- Persistent working sessions across tool calls

Cloud storage integrations (OneDrive, SharePoint) are available as **optional convenience layers** when environment credentials are configured. They are not required for core functionality.

---

## Architecture

```text
SimTheory.ai (MCP Client)
    │
    ▼  JSON-RPC over HTTP POST
┌──────────────────────────────────────────────┐
│  Power Interpreter (Railway)                 │
│                                              │
│  ┌──────────────┐  ┌──────────────────────┐  │
│  │ MCP Server   │  │ FastAPI Routes       │  │
│  │ (base tools  │──│ /api/execute         │  │
│  │  + optional  │  │ /api/data/load       │  │
│  │  cloud       │  │ /api/files/*         │  │
│  │  integrations│  │ /api/jobs/*          │  │
│  │  )           │  │                      │  │
│  └──────────────┘  └──────────────────────┘  │
│         │                    │                │
│         ▼                    ▼                │
│  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Python       │  │ PostgreSQL           │  │
│  │ Kernel       │  │ (datasets, files,    │  │
│  │ (persistent  │  │  jobs, metadata)     │  │
│  │  sessions)   │  │                      │  │
│  └──────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────┘
```

---

## MCP Tools

### Base Tools (always available)

| Tool | Description |
|------|-------------|
| `execute_code` | Run Python code in a persistent sandbox kernel (sync, <60s) |
| `submit_job` | Submit long-running code for async execution (up to 30 min) |
| `get_job_status` | Check async job progress |
| `get_job_result` | Retrieve completed job output |
| `fetch_from_url` | Download file from any HTTPS URL into sandbox |
| `upload_file` | Upload a file via base64 encoding (<10MB) |
| `fetch_file` | Download file from URL (legacy helper) |
| `list_files` | List files in the sandbox |
| `load_dataset` | Load data file into PostgreSQL — auto-detects format |
| `query_dataset` | Execute SQL SELECT queries against loaded datasets |
| `list_datasets` | List all datasets in PostgreSQL |
| `create_session` | Create isolated workspace session |

### Optional Integrations

Microsoft 365 (OneDrive / SharePoint) tools are available when `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, and `MICROSOFT_TENANT_ID` environment variables are configured. These are not required for core operation.

---

## Supported Data Formats

The `load_dataset` tool auto-detects file format from the file extension:

| Format | Extensions | Reader | Notes |
|--------|-----------|--------|-------|
| **CSV** | `.csv`, `.tsv`, `.txt` | `pd.read_csv()` | Chunked loading for large files |
| **Excel** | `.xlsx`, `.xls`, `.xlsm`, `.xlsb` | `pd.read_excel()` | Full read, then chunked insert |
| **PDF** | `.pdf` | `pdfplumber` | Extracts tabular data from PDF pages |
| **JSON** | `.json` | `pd.read_json()` + `json_normalize` | Array of objects or nested JSON |
| **Parquet** | `.parquet`, `.pq` | `pd.read_parquet()` | Columnar format, very fast |

All formats are loaded into PostgreSQL in 50K-row chunks with automatic indexing on date and ID columns. Handles **1.5M+ rows** efficiently.

### Typical Workflow

```text
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
| `timedelta` | datetime.timedelta | v2.8.4 |
| `timezone` | datetime.timezone | v2.8.4 |
| `date` | datetime.date | v2.8.4 |
| `json`, `csv`, `math`, `re`, `io`, `copy` | stdlib modules | v1.0+ |
| `collections`, `itertools`, `functools` | stdlib modules | v1.0 |
| `statistics`, `hashlib`, `base64` | stdlib modules | v1.0 |
| `Decimal`, `Fraction`, `Path` | stdlib classes | v1.0 |

### Lazy-loaded Libraries (loaded on first import)

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
| reportlab | | platypus, pdfgen sub-modules |
| requests | | HTTP client |
| warnings | | utility support |
| abc | | abstract base classes |
| enum | | enum support |
| weakref | | weak reference support |
| python-docx | `Document` | .docx document support |
| tabulate, textwrap, string, struct | | |
| decimal, fractions, random, time, calendar | | |
| pprint, dataclasses, typing, pathlib, os | | |
| urllib, shutil, glob | | |

### Path Safety

- **Session prefix stripping**: `default/file.csv` → `file.csv` when cwd is already the session dir
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
- Uvicorn ASGI server

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `API_KEY` | Yes | API key for protected endpoints |
| `RAILWAY_PUBLIC_DOMAIN` | Auto | Set by Railway for public URLs |
| `MICROSOFT_CLIENT_ID` | No | Optional Microsoft integration |
| `MICROSOFT_CLIENT_SECRET` | No | Optional Microsoft integration |
| `MICROSOFT_TENANT_ID` | No | Optional Microsoft integration |

### Configuration Defaults

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

```text
power-interpreter/
├── app/
│   ├── version.py             # Single source of truth for version
│   ├── main.py                # FastAPI app + MCP JSON-RPC handler
│   ├── mcp_server.py          # MCP tool definitions
│   ├── config.py              # Settings and environment config
│   ├── auth.py                # API key authentication
│   ├── database.py            # PostgreSQL connection management
│   ├── models.py              # SQLAlchemy models
│   ├── engine/
│   │   ├── data_manager.py    # Universal data loading (CSV/Excel/PDF/JSON/Parquet)
│   │   ├── executor.py        # Sandboxed Python execution
│   │   ├── file_manager.py    # Sandbox file management
│   │   ├── job_manager.py     # Async job queue
│   │   └── kernel_manager.py  # Persistent Python kernel sessions
│   └── routes/
│       ├── data.py
│       ├── execute.py
│       ├── files.py
│       ├── health.py
│       ├── jobs.py
│       └── sessions.py
├── Dockerfile
├── requirements.txt
├── start.py                   # Railway/Uvicorn startup entrypoint
└── README.md
```

---

## Version History

| Version | Date | Component | Changes |
|---------|------|-----------|---------|
| **v2.9.1** | 2026-03-07 | main, start, docs | Centralized version constant; startup consistency; `datetime.utcnow()` replaced; docs refresh |
| **v2.9.0** | 2026-03 | mcp_server | Trimmed all tool descriptions for token optimization (~57% reduction) |
| **v2.8.6** | 2026-03 | all | Version unification across all files |
| **v2.8.5** | 2026-03 | executor | Expanded import allowlist (warnings, abc, enum, weakref, Document alias) |
| **v2.8.4** | 2026-02-28 | executor, main | datetime convenience aliases (timedelta, timezone, date) |
| **v2.8.3** | 2026-02-22 | executor | /app/sandbox_data added to allowed read paths |
| **v2.8.2** | 2026-02-22 | executor | Read-only upload access for files outside sandbox |
| **v2.8.1** | 2026-02-22 | executor | /tmp/ path interception and redirect to sandbox |
| **v2.8.0** | 2026-02-22 | executor | Defensive path normalization (doubled session prefix) |
| **v2.7.0** | 2026-02-21 | executor | reportlab + matplotlib PDF backend allowlisted |
| **v2.6** | 2026-02-20 | executor | Critical fix: matplotlib.pyplot alias override bug |
| **v1.9.2** | 2026-02 | Microsoft auth | Token persistence rewrite (SQLAlchemy) |
| **v1.9.0** | 2026-02 | Microsoft integration | OneDrive + SharePoint integration |
| **v1.8.2** | 2026-02-23 | mcp_server | load_dataset description updated for universal format |
| **v1.8.1** | 2026-02-23 | main, mcp_server | Chart serve route + base64 stdout regex fallback |
| **v1.7.2** | 2026-02-22 | mcp_server | fetch_from_url route fix |
| **v1.2.0** | 2026-02-15 | all | Initial working version |

---

## Author

Built by **Kaffer AI** for **Timothy Escamilla**.

Designed to support practical AI workflows across analytics, operations, and personal productivity.
