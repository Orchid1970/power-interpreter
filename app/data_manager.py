"""Power Interpreter - Data Manager

Manages large datasets in PostgreSQL:
- Load CSV/Excel into PostgreSQL tables (1.5M+ rows)
- Query datasets with SQL
- Export datasets back to files
- Track dataset metadata

This is the key component for handling large data that won't fit in memory.
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


class DataManager:
    """Manages large datasets in PostgreSQL"""
    
    # Prefix for dynamic data tables to avoid conflicts
    TABLE_PREFIX = "data_"
    
    # Chunk size for loading large files
    LOAD_CHUNK_SIZE = 50000  # 50K rows at a time
    
    async def load_csv(
        self,
        file_path: str,
        dataset_name: str,
        session_id: str = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        dtypes: Dict = None,
        parse_dates: List[str] = None
    ) -> Dict:
        """Load a CSV file into PostgreSQL
        
        Handles files with 1.5M+ rows by loading in chunks.
        
        Args:
            file_path: Path to CSV file (can be relative - will be resolved)
            dataset_name: Logical name for the dataset
            session_id: Session to associate with
            delimiter: CSV delimiter
            encoding: File encoding
            dtypes: Column type overrides
            parse_dates: Columns to parse as dates
        
        Returns:
            Dataset metadata dict
        """
        # ============================================================
        # FIX: Resolve file path against sandbox directory
        # The MCP client passes relative paths, but the file lives
        # in /app/sandbox_data/default/ or similar.
        # ============================================================
        resolved_path = resolve_file_path(file_path)
        logger.info(f"Resolved '{file_path}' -> '{resolved_path}'")
        
        dataset_id = str(uuid.uuid4())
        table_name = f"{self.TABLE_PREFIX}{dataset_id.replace('-', '_')}"
        
        logger.info(f"Loading CSV into PostgreSQL: {resolved_path} -> {table_name}")
        
        # Read CSV in chunks and load into PostgreSQL
        engine = get_engine()
        total_rows = 0
        columns_info = None
        
        try:
            # First pass: read a small sample to get schema
            sample_df = pd.read_csv(
                resolved_path, 
                nrows=100, 
                delimiter=delimiter,
                encoding=encoding,
                dtype=dtypes,
                parse_dates=parse_dates
            )
            
            columns_info = [
                {'name': col, 'dtype': str(sample_df[col].dtype), 'nullable': True}
                for col in sample_df.columns
            ]
            
            # Load in chunks using pandas + SQLAlchemy
            chunk_iter = pd.read_csv(
                resolved_path,
                chunksize=self.LOAD_CHUNK_SIZE,
                delimiter=delimiter,
                encoding=encoding,
                dtype=dtypes,
                parse_dates=parse_dates,
                low_memory=False
            )
            
            # We need sync engine for pandas to_sql
            from sqlalchemy import create_engine
            from app.config import settings as cfg
            sync_engine = create_engine(cfg.sync_database_url)
            
            for i, chunk in enumerate(chunk_iter):
                if_exists = 'replace' if i == 0 else 'append'
                chunk.to_sql(
                    table_name, 
                    sync_engine, 
                    if_exists=if_exists, 
                    index=False,
                    method='multi'
                )
                total_rows += len(chunk)
                logger.info(f"  Loaded chunk {i+1}: {total_rows} rows so far")
            
            sync_engine.dispose()
            
            # Get table size
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(f"SELECT pg_total_relation_size('{table_name}')")
                )
                size_bytes = result.scalar() or 0
            
            # Create indexes on common column types
            await self._create_auto_indexes(table_name, columns_info)
            
            # Save dataset metadata
            factory = get_session_factory()
            async with factory() as session:
                dataset = Dataset(
                    id=uuid.UUID(dataset_id),
                    session_id=uuid.UUID(session_id) if session_id else None,
                    name=dataset_name,
                    table_name=table_name,
                    description=f"Loaded from {Path(file_path).name}",
                    row_count=total_rows,
                    column_count=len(columns_info),
                    columns=columns_info,
                    source_filename=Path(file_path).name,
                    size_bytes=size_bytes,
                )
                session.add(dataset)
                await session.commit()
            
            logger.info(
                f"Dataset '{dataset_name}' loaded: {total_rows} rows, "
                f"{len(columns_info)} columns, {size_bytes/1024/1024:.1f} MB"
            )
            
            # Get preview
            preview = sample_df.head(5).to_dict(orient='records')
            
            return {
                'dataset_id': dataset_id,
                'name': dataset_name,
                'table_name': table_name,
                'row_count': total_rows,
                'column_count': len(columns_info),
                'columns': columns_info,
                'size_mb': round(size_bytes / 1024 / 1024, 2),
                'preview': preview,
                'status': 'loaded'
            }
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            # Cleanup on failure
            try:
                async with engine.connect() as conn:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    await conn.commit()
            except Exception:
                pass
            raise
    
    async def query_dataset(
        self,
        sql: str,
        params: Dict = None,
        limit: int = 1000,
        offset: int = 0
    ) -> Dict:
        """Execute SQL query against datasets
        
        Args:
            sql: SQL query (SELECT only for safety)
            params: Query parameters
            limit: Max rows to return
            offset: Row offset for pagination
        
        Returns:
            Query results with metadata
        """
        # Safety check - only allow SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed. Use load_csv() to modify data.")
        
        # Block dangerous operations
        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE', 'GRANT']
        for keyword in dangerous:
            if keyword in sql_upper and keyword != 'CREATE':  # Allow CREATE in subqueries
                raise ValueError(f"Operation '{keyword}' is not allowed in queries.")
        
        # Add pagination if not present
        if 'LIMIT' not in sql_upper:
            sql = f"{sql} LIMIT {limit} OFFSET {offset}"
        
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = list(result.keys())
            
            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in rows]
            
            # Serialize datetime objects
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
                query = query.where(Dataset.session_id == uuid.UUID(session_id))
            
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
            
            # Drop the actual table
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                await conn.commit()
            
            # Delete metadata
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
                
                # Index date columns
                if 'date' in dtype.lower() or 'date' in col_name.lower():
                    try:
                        idx_name = f"idx_{table_name}_{col_name}".replace(' ', '_')[:63]
                        await conn.execute(
                            text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col_name}")')
                        )
                    except Exception:
                        pass
                
                # Index ID-like columns
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
