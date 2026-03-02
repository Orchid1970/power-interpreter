"""Power Interpreter - Data Manager

Manages large datasets in PostgreSQL:
- Load CSV/Excel/PDF/JSON/Parquet into PostgreSQL tables (1.5M+ rows)
- Query datasets with SQL
- Export datasets back to files
- Track dataset metadata

Supported formats:
  .csv, .tsv, .txt  -- pandas read_csv
  .xlsx, .xls, .xlsm, .xlsb -- pandas read_excel (openpyxl)
  .pdf -- pdfplumber table extraction
  .json -- pandas read_json
  .parquet, .pq -- pandas read_parquet
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from app.config import settings
from app.models import Dataset
from app.database import get_engine, get_session_factory

logger = logging.getLogger(__name__)

# ============================================================
# Format detection
# ============================================================
FORMAT_CSV = "csv"
FORMAT_EXCEL = "excel"
FORMAT_PDF = "pdf"
FORMAT_JSON = "json"
FORMAT_PARQUET = "parquet"

EXTENSION_MAP = {
    '.csv': FORMAT_CSV,
    '.tsv': FORMAT_CSV,
    '.txt': FORMAT_CSV,
    '.xlsx': FORMAT_EXCEL,
    '.xls': FORMAT_EXCEL,
    '.xlsm': FORMAT_EXCEL,
    '.xlsb': FORMAT_EXCEL,
    '.pdf': FORMAT_PDF,
    '.json': FORMAT_JSON,
    '.parquet': FORMAT_PARQUET,
    '.pq': FORMAT_PARQUET,
}


def detect_format(file_path: str) -> str:
    """Detect file format from extension"""
    ext = Path(file_path).suffix.lower()
    fmt = EXTENSION_MAP.get(ext)
    if not fmt:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
        )
    return fmt


def _safe_parse_uuid(value: str) -> Optional[uuid.UUID]:
    """Safely parse a string as UUID, returning None for non-UUID values.
    
    The MCP server may pass session_id="default" when no explicit session
    has been created. This helper prevents uuid.UUID("default") from
    raising 'badly formed hexadecimal UUID string'.
    """
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        logger.debug(f"session_id '{value}' is not a valid UUID, treating as None")
        return None


def resolve_file_path(file_path: str) -> str:
    """Resolve a file path by searching multiple locations.
    
    The MCP client may pass paths like:
      - 'intercompany_accounts.csv'
      - 'default/intercompany_accounts.csv'
    
    But the actual file lives in the sandbox directory.
    This function tries multiple candidate paths and returns
    the first one that exists.
    """
    from app.config import settings as cfg
    
    raw_path = Path(file_path)
    filename = raw_path.name
    
    # Build list of candidate paths to try
    candidates = [
        raw_path,                                          # As-is (absolute or relative)
        cfg.SANDBOX_DIR / file_path,                       # /app/sandbox_data/default/file.csv
        cfg.SANDBOX_DIR / "default" / filename,            # /app/sandbox_data/default/file.csv
        cfg.SANDBOX_DIR / filename,                        # /app/sandbox_data/file.csv
        cfg.UPLOAD_DIR / file_path,                        # /app/uploads/default/file.csv
        cfg.UPLOAD_DIR / filename,                         # /app/uploads/file.csv
    ]
    
    # Also walk the sandbox directory for the file
    sandbox_root = cfg.SANDBOX_DIR
    if sandbox_root.exists():
        for root, dirs, files in os.walk(sandbox_root):
            if filename in files:
                candidates.append(Path(root) / filename)
    
    # Try each candidate
    for candidate in candidates:
        resolved = Path(candidate)
        if resolved.exists() and resolved.is_file():
            logger.info(f"Resolved file path: {file_path} -> {resolved}")
            return str(resolved)
    
    # Log all candidates for debugging
    logger.error(f"Could not resolve file path: {file_path}")
    logger.error(f"Tried candidates: {[str(c) for c in candidates]}")
    logger.error(f"SANDBOX_DIR={cfg.SANDBOX_DIR}, exists={cfg.SANDBOX_DIR.exists()}")
    if cfg.SANDBOX_DIR.exists():
        try:
            all_files = list(cfg.SANDBOX_DIR.rglob('*'))
            logger.error(f"Files in sandbox: {[str(f) for f in all_files[:20]]}")
        except Exception:
            pass
    
    raise FileNotFoundError(
        f"File not found: {file_path}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


# ============================================================
# Format-specific readers
# ============================================================

def _read_csv_sample(file_path: str, delimiter: str, encoding: str,
                     dtypes: Dict = None, parse_dates: List[str] = None) -> pd.DataFrame:
    """Read a small sample from a CSV file"""
    return pd.read_csv(
        file_path, nrows=100, delimiter=delimiter,
        encoding=encoding, dtype=dtypes, parse_dates=parse_dates
    )


def _read_csv_chunks(file_path: str, chunk_size: int, delimiter: str,
                     encoding: str, dtypes: Dict = None, parse_dates: List[str] = None):
    """Yield chunks from a CSV file"""
    return pd.read_csv(
        file_path, chunksize=chunk_size, delimiter=delimiter,
        encoding=encoding, dtype=dtypes, parse_dates=parse_dates,
        low_memory=False
    )


def _read_excel_full(file_path: str, sheet_name=None,
                     dtypes: Dict = None, parse_dates: List[str] = None) -> pd.DataFrame:
    """Read an entire Excel file (openpyxl doesn't support chunked reading)"""
    kwargs = {'dtype': dtypes}
    if parse_dates:
        kwargs['parse_dates'] = parse_dates
    if sheet_name is not None:
        kwargs['sheet_name'] = sheet_name

    return pd.read_excel(file_path, **kwargs)


def _read_pdf_tables(file_path: str, pages: str = None) -> pd.DataFrame:
    """Extract tables from a PDF using pdfplumber.
    
    Concatenates all tables found across the specified pages.
    If no tables are found, raises ValueError.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ValueError(
            "pdfplumber is not installed. Cannot extract tables from PDF. "
            "Install with: pip install pdfplumber"
        )

    all_tables = []

    with pdfplumber.open(file_path) as pdf:
        # Determine which pages to process
        if pages and pages.lower() != 'all':
            page_nums = [int(p.strip()) - 1 for p in pages.split(',')]
            target_pages = [pdf.pages[i] for i in page_nums if i < len(pdf.pages)]
        else:
            target_pages = pdf.pages

        for page in target_pages:
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    # First row is header
                    header = [str(h).strip() if h else f"col_{i}"
                              for i, h in enumerate(table[0])]
                    rows = table[1:]
                    df = pd.DataFrame(rows, columns=header)
                    all_tables.append(df)

    if not all_tables:
        raise ValueError(
            f"No tables found in PDF: {file_path}. "
            f"Scanned {len(target_pages)} page(s). "
            f"If the PDF contains unstructured text, use execute_code with "
            f"pdfplumber directly to extract specific content."
        )

    # Concatenate all tables
    combined = pd.concat(all_tables, ignore_index=True)

    # Try to convert numeric columns
    for col in combined.columns:
        try:
            combined[col] = pd.to_numeric(combined[col])
        except (ValueError, TypeError):
            pass

    logger.info(f"Extracted {len(all_tables)} table(s) from PDF: "
                f"{len(combined)} rows, {len(combined.columns)} columns")
    return combined


def _read_json_full(file_path: str) -> pd.DataFrame:
    """Read a JSON file into a DataFrame.
    
    Supports:
      - Array of objects: [{"a": 1}, {"a": 2}]
      - Records format
      - Nested JSON (will be flattened)
    """
    import json as json_lib

    # Try pandas read_json first
    try:
        return pd.read_json(file_path)
    except ValueError:
        pass

    # Fallback: load raw and normalize
    with open(file_path, 'r') as f:
        raw = json_lib.load(f)

    if isinstance(raw, list):
        return pd.json_normalize(raw)
    elif isinstance(raw, dict):
        # Check if it's a dict of lists (column-oriented)
        if all(isinstance(v, list) for v in raw.values()):
            return pd.DataFrame(raw)
        # Single record or nested
        return pd.json_normalize(raw)
    else:
        raise ValueError(f"Cannot parse JSON into tabular format. "
                         f"Top-level type: {type(raw).__name__}")


def _read_parquet_full(file_path: str) -> pd.DataFrame:
    """Read a Parquet file"""
    return pd.read_parquet(file_path)


# ============================================================
# DataManager class
# ============================================================

class DataManager:
    """Manages large datasets in PostgreSQL"""

    # Prefix for dynamic data tables to avoid conflicts
    TABLE_PREFIX = "data_"

    # Chunk size for loading large files
    LOAD_CHUNK_SIZE = 50000  # 50K rows at a time

    async def load_data(
        self,
        file_path: str,
        dataset_name: str,
        session_id: str = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        dtypes: Dict = None,
        parse_dates: List[str] = None,
        sheet_name: str = None,
        pdf_pages: str = None,
    ) -> Dict:
        """Load any supported file format into PostgreSQL.
        
        Detects format from file extension and uses the appropriate reader.
        Handles large files via chunked loading where possible.
        
        Args:
            file_path: Path to file (will be resolved against sandbox)
            dataset_name: Logical name for the dataset
            session_id: Session to associate with
            delimiter: CSV delimiter (CSV only)
            encoding: File encoding (CSV only)
            dtypes: Column type overrides
            parse_dates: Columns to parse as dates
            sheet_name: Excel sheet name (Excel only)
            pdf_pages: PDF pages to extract (PDF only)
        
        Returns:
            Dataset metadata dict
        """
        # Resolve file path
        resolved_path = resolve_file_path(file_path)
        logger.info(f"Resolved '{file_path}' -> '{resolved_path}'")
        file_path = resolved_path

        # Detect format
        fmt = detect_format(file_path)
        logger.info(f"Detected format: {fmt} for {Path(file_path).name}")

        dataset_id = str(uuid.uuid4())
        table_name = f"{self.TABLE_PREFIX}{dataset_id.replace('-', '_')}"

        engine = get_engine()
        total_rows = 0
        columns_info = None
        sample_df = None

        try:
            # ============================================
            # Read data based on detected format
            # ============================================

            if fmt == FORMAT_CSV:
                # --- CSV: chunked loading ---
                sample_df = _read_csv_sample(
                    file_path, delimiter, encoding, dtypes, parse_dates
                )
                columns_info = self._extract_columns_info(sample_df)

                sync_engine = self._get_sync_engine()
                chunk_iter = _read_csv_chunks(
                    file_path, self.LOAD_CHUNK_SIZE,
                    delimiter, encoding, dtypes, parse_dates
                )
                for i, chunk in enumerate(chunk_iter):
                    if_exists = 'replace' if i == 0 else 'append'
                    chunk.to_sql(table_name, sync_engine,
                                 if_exists=if_exists, index=False, method='multi')
                    total_rows += len(chunk)
                    logger.info(f"  CSV chunk {i+1}: {total_rows} rows")
                sync_engine.dispose()

            elif fmt == FORMAT_EXCEL:
                # --- Excel: read full, then chunk into Postgres ---
                full_df = _read_excel_full(file_path, sheet_name, dtypes, parse_dates)
                columns_info = self._extract_columns_info(full_df)
                sample_df = full_df.head(100)

                sync_engine = self._get_sync_engine()
                total_rows = self._load_dataframe_chunked(
                    full_df, table_name, sync_engine, "Excel"
                )
                sync_engine.dispose()

            elif fmt == FORMAT_PDF:
                # --- PDF: extract tables with pdfplumber ---
                full_df = _read_pdf_tables(file_path, pdf_pages)
                columns_info = self._extract_columns_info(full_df)
                sample_df = full_df.head(100)

                sync_engine = self._get_sync_engine()
                total_rows = self._load_dataframe_chunked(
                    full_df, table_name, sync_engine, "PDF"
                )
                sync_engine.dispose()

            elif fmt == FORMAT_JSON:
                # --- JSON: read full, then chunk ---
                full_df = _read_json_full(file_path)
                columns_info = self._extract_columns_info(full_df)
                sample_df = full_df.head(100)

                sync_engine = self._get_sync_engine()
                total_rows = self._load_dataframe_chunked(
                    full_df, table_name, sync_engine, "JSON"
                )
                sync_engine.dispose()

            elif fmt == FORMAT_PARQUET:
                # --- Parquet: read full, then chunk ---
                full_df = _read_parquet_full(file_path)
                columns_info = self._extract_columns_info(full_df)
                sample_df = full_df.head(100)

                sync_engine = self._get_sync_engine()
                total_rows = self._load_dataframe_chunked(
                    full_df, table_name, sync_engine, "Parquet"
                )
                sync_engine.dispose()

            else:
                raise ValueError(f"Unsupported format: {fmt}")

            # ============================================
            # Post-load: indexes, metadata, response
            # ============================================

            # Get table size
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(f"SELECT pg_total_relation_size('{table_name}')")
                )
                size_bytes = result.scalar() or 0

            # Create indexes on common column types
            await self._create_auto_indexes(table_name, columns_info)

            # Save dataset metadata
            parsed_session_id = _safe_parse_uuid(session_id)
            factory = get_session_factory()
            async with factory() as session:
                dataset = Dataset(
                    id=uuid.UUID(dataset_id),
                    session_id=parsed_session_id,
                    name=dataset_name,
                    table_name=table_name,
                    description=f"Loaded from {Path(file_path).name} ({fmt})",
                    row_count=total_rows,
                    column_count=len(columns_info),
                    columns=columns_info,
                    source_filename=Path(file_path).name,
                    size_bytes=size_bytes,
                )
                session.add(dataset)
                await session.commit()

            logger.info(
                f"Dataset '{dataset_name}' loaded ({fmt}): {total_rows} rows, "
                f"{len(columns_info)} columns, {size_bytes/1024/1024:.1f} MB"
            )

            # Preview
            preview = sample_df.head(5).to_dict(orient='records')

            return {
                'dataset_id': dataset_id,
                'name': dataset_name,
                'table_name': table_name,
                'format': fmt,
                'row_count': total_rows,
                'column_count': len(columns_info),
                'columns': columns_info,
                'size_mb': round(size_bytes / 1024 / 1024, 2),
                'preview': preview,
                'status': 'loaded'
            }

        except Exception as e:
            logger.error(f"Failed to load {fmt} data: {e}", exc_info=True)
            # Cleanup on failure
            try:
                async with engine.connect() as conn:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    await conn.commit()
            except Exception:
                pass
            raise

    # ============================================
    # Backwards compatibility alias
    # ============================================
    async def load_csv(self, file_path: str, dataset_name: str,
                       session_id: str = None, delimiter: str = ',',
                       encoding: str = 'utf-8', dtypes: Dict = None,
                       parse_dates: List[str] = None) -> Dict:
        """Legacy method -- routes through load_data with format auto-detection."""
        return await self.load_data(
            file_path=file_path,
            dataset_name=dataset_name,
            session_id=session_id,
            delimiter=delimiter,
            encoding=encoding,
            dtypes=dtypes,
            parse_dates=parse_dates,
        )

    # ============================================
    # Helpers
    # ============================================

    def _get_sync_engine(self):
        """Create a sync SQLAlchemy engine for pandas to_sql"""
        from sqlalchemy import create_engine
        from app.config import settings as cfg
        return create_engine(cfg.sync_database_url)

    @staticmethod
    def _extract_columns_info(df: pd.DataFrame) -> List[Dict]:
        """Extract column metadata from a DataFrame"""
        return [
            {'name': col, 'dtype': str(df[col].dtype), 'nullable': True}
            for col in df.columns
        ]

    def _load_dataframe_chunked(self, df: pd.DataFrame, table_name: str,
                                 sync_engine, format_label: str) -> int:
        """Load a DataFrame into Postgres in chunks. Returns total rows."""
        total_rows = 0
        for i in range(0, len(df), self.LOAD_CHUNK_SIZE):
            chunk = df.iloc[i:i + self.LOAD_CHUNK_SIZE]
            if_exists = 'replace' if i == 0 else 'append'
            chunk.to_sql(table_name, sync_engine,
                         if_exists=if_exists, index=False, method='multi')
            total_rows += len(chunk)
            chunk_num = i // self.LOAD_CHUNK_SIZE + 1
            logger.info(f"  {format_label} chunk {chunk_num}: {total_rows} rows")
        return total_rows

    # ============================================
    # Query, list, info, drop
    # ============================================

    async def query_dataset(
        self,
        sql: str,
        params: Dict = None,
        limit: int = 1000,
        offset: int = 0
    ) -> Dict:
        """Execute SQL query against datasets"""
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed. Use load_data() to modify data.")

        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'GRANT']
        for keyword in dangerous:
            if keyword in sql_upper:
                raise ValueError(f"Operation '{keyword}' is not allowed in queries.")

        if 'LIMIT' not in sql_upper:
            sql = f"{sql} LIMIT {limit} OFFSET {offset}"

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = list(result.keys())

            data = [dict(zip(columns, row)) for row in rows]

            for row in data:
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row[key] = value.isoformat()

            return {
                'columns': columns,
                'data': data,
                'row_count': len(data),
                'limit': limit,
                'offset': offset,
                'has_more': len(data) == limit
            }

    async def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get dataset metadata by name"""
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            dataset = result.scalar_one_or_none()

            if not dataset:
                return None

            return {
                'dataset_id': str(dataset.id),
                'name': dataset.name,
                'table_name': dataset.table_name,
                'description': dataset.description,
                'row_count': dataset.row_count,
                'column_count': dataset.column_count,
                'columns': dataset.columns,
                'size_mb': round((dataset.size_bytes or 0) / 1024 / 1024, 2),
                'source_filename': dataset.source_filename,
                'created_at': dataset.created_at.isoformat(),
            }

    async def list_datasets(self, session_id: str = None) -> List[Dict]:
        """List all datasets"""
        factory = get_session_factory()
        async with factory() as session:
            query = select(Dataset).order_by(Dataset.created_at.desc())
            if session_id:
                parsed_id = _safe_parse_uuid(session_id)
                if parsed_id:
                    query = query.where(Dataset.session_id == parsed_id)

            result = await session.execute(query)
            datasets = result.scalars().all()

            return [{
                'dataset_id': str(d.id),
                'name': d.name,
                'row_count': d.row_count,
                'column_count': d.column_count,
                'size_mb': round((d.size_bytes or 0) / 1024 / 1024, 2),
                'created_at': d.created_at.isoformat(),
            } for d in datasets]

    async def drop_dataset(self, dataset_name: str) -> bool:
        """Drop a dataset (table + metadata)"""
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            dataset = result.scalar_one_or_none()

            if not dataset:
                return False

            table_name = dataset.table_name

            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                await conn.commit()

            await session.delete(dataset)
            await session.commit()

            logger.info(f"Dataset '{dataset_name}' dropped (table: {table_name})")
            return True

    async def _create_auto_indexes(self, table_name: str, columns: List[Dict]):
        """Create indexes on date and ID-like columns for faster queries"""
        engine = get_engine()
        async with engine.connect() as conn:
            for col_info in columns:
                col_name = col_info['name']
                dtype = col_info['dtype']

                if 'date' in dtype.lower() or 'date' in col_name.lower():
                    try:
                        idx_name = f"idx_{table_name}_{col_name}".replace(' ', '_')[:63]
                        await conn.execute(
                            text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col_name}")')
                        )
                    except Exception:
                        pass

                if col_name.lower().endswith('_id') or col_name.lower() == 'id':
                    try:
                        idx_name = f"idx_{table_name}_{col_name}".replace(' ', '_')[:63]
                        await conn.execute(
                            text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col_name}")')
                        )
                    except Exception:
                        pass

            await conn.commit()


# Singleton
data_manager = DataManager()
