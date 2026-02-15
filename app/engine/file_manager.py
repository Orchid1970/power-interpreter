"""Power Interpreter - File Manager

Manages file uploads, downloads, and sandbox file operations:
- Chunked file uploads for large files
- File metadata tracking in PostgreSQL
- CSV/Excel auto-detection and preview
- Persistent storage on Railway volume
"""

import os
import uuid
import hashlib
import aiofiles
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, BinaryIO
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import File, FileType
from app.database import get_session_factory

logger = logging.getLogger(__name__)


class FileManager:
    """Manages files in the sandbox environment"""
    
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.sandbox_dir = settings.SANDBOX_DIR
        self.temp_dir = settings.TEMP_DIR
        
        # Ensure directories exist
        for d in [self.upload_dir, self.sandbox_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(
        self,
        filename: str,
        content: bytes,
        session_id: str = None,
        file_type: FileType = FileType.UPLOAD
    ) -> Dict:
        """Upload a file to the sandbox
        
        Args:
            filename: Original filename
            content: File content as bytes
            session_id: Session to associate with
            file_type: Type of file
        
        Returns:
            File metadata dict
        """
        file_id = str(uuid.uuid4())
        
        # Determine storage path
        if session_id:
            storage_dir = self.sandbox_dir / session_id
        else:
            storage_dir = self.upload_dir
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use original filename (sanitized)
        safe_filename = self._sanitize_filename(filename)
        storage_path = storage_dir / safe_filename
        
        # Handle name conflicts
        if storage_path.exists():
            stem = storage_path.stem
            suffix = storage_path.suffix
            counter = 1
            while storage_path.exists():
                storage_path = storage_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        # Write file
        async with aiofiles.open(storage_path, 'wb') as f:
            await f.write(content)
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Detect mime type
        mime_type = self._detect_mime_type(safe_filename)
        
        # Get file metadata (row count for CSVs, etc.)
        file_meta = await self._analyze_file(storage_path, mime_type)
        
        # Save to database
        factory = get_session_factory()
        async with factory() as session:
            file_record = File(
                id=uuid.UUID(file_id),
                session_id=uuid.UUID(session_id) if session_id else None,
                filename=safe_filename,
                original_filename=filename,
                file_type=file_type,
                mime_type=mime_type,
                file_size=len(content),
                storage_path=str(storage_path),
                checksum=checksum,
                row_count=file_meta.get('row_count'),
                column_count=file_meta.get('column_count'),
                columns=file_meta.get('columns'),
                preview=file_meta.get('preview'),
            )
            session.add(file_record)
            await session.commit()
        
        logger.info(f"File uploaded: {safe_filename} ({len(content)} bytes) -> {storage_path}")
        
        return {
            'file_id': file_id,
            'filename': safe_filename,
            'original_filename': filename,
            'file_type': file_type.value,
            'mime_type': mime_type,
            'file_size': len(content),
            'storage_path': str(storage_path),
            'checksum': checksum,
            **file_meta
        }
    
    async def get_file(self, file_id: str) -> Optional[Dict]:
        """Get file metadata by ID"""
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(File).where(File.id == uuid.UUID(file_id))
            )
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                return None
            
            return {
                'file_id': str(file_record.id),
                'filename': file_record.filename,
                'original_filename': file_record.original_filename,
                'file_type': file_record.file_type.value,
                'mime_type': file_record.mime_type,
                'file_size': file_record.file_size,
                'storage_path': file_record.storage_path,
                'checksum': file_record.checksum,
                'row_count': file_record.row_count,
                'column_count': file_record.column_count,
                'columns': file_record.columns,
                'preview': file_record.preview,
                'created_at': file_record.created_at.isoformat(),
            }
    
    async def download_file(self, file_id: str) -> Optional[tuple]:
        """Get file content for download
        
        Returns:
            Tuple of (content_bytes, filename, mime_type) or None
        """
        file_info = await self.get_file(file_id)
        if not file_info:
            return None
        
        storage_path = Path(file_info['storage_path'])
        if not storage_path.exists():
            return None
        
        async with aiofiles.open(storage_path, 'rb') as f:
            content = await f.read()
        
        return (content, file_info['filename'], file_info['mime_type'])
    
    async def list_files(
        self, 
        session_id: str = None,
        file_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """List files with optional filters"""
        factory = get_session_factory()
        async with factory() as session:
            query = select(File).order_by(File.created_at.desc()).limit(limit)
            
            if session_id:
                query = query.where(File.session_id == uuid.UUID(session_id))
            if file_type:
                query = query.where(File.file_type == FileType(file_type))
            
            result = await session.execute(query)
            files = result.scalars().all()
            
            return [{
                'file_id': str(f.id),
                'filename': f.filename,
                'file_type': f.file_type.value,
                'file_size': f.file_size,
                'row_count': f.row_count,
                'column_count': f.column_count,
                'created_at': f.created_at.isoformat(),
            } for f in files]
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete a file from storage and database"""
        file_info = await self.get_file(file_id)
        if not file_info:
            return False
        
        # Delete from disk
        storage_path = Path(file_info['storage_path'])
        if storage_path.exists():
            storage_path.unlink()
        
        # Delete from database
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                delete(File).where(File.id == uuid.UUID(file_id))
            )
            await session.commit()
        
        logger.info(f"File deleted: {file_info['filename']}")
        return True
    
    async def list_sandbox_files(self, session_id: str = "default") -> List[Dict]:
        """List files in a session's sandbox directory"""
        session_dir = self.sandbox_dir / session_id
        if not session_dir.exists():
            return []
        
        files = []
        for path in sorted(session_dir.rglob('*')):
            if path.is_file():
                rel_path = path.relative_to(session_dir)
                stat = path.stat()
                files.append({
                    'path': str(rel_path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': self._detect_mime_type(path.name),
                })
        
        return files
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        # Remove dangerous characters
        filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        return filename or 'unnamed_file'
    
    def _detect_mime_type(self, filename: str) -> str:
        """Detect MIME type from filename"""
        ext = Path(filename).suffix.lower()
        mime_map = {
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.py': 'text/x-python',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.html': 'text/html',
            '.md': 'text/markdown',
        }
        return mime_map.get(ext, 'application/octet-stream')
    
    async def _analyze_file(self, path: Path, mime_type: str) -> Dict:
        """Analyze file to extract metadata"""
        meta = {}
        
        try:
            if mime_type == 'text/csv':
                import pandas as pd
                # Read just enough to get metadata
                df = pd.read_csv(path, nrows=5)
                full_count = sum(1 for _ in open(path)) - 1  # Subtract header
                
                meta['row_count'] = full_count
                meta['column_count'] = len(df.columns)
                meta['columns'] = [
                    {'name': col, 'dtype': str(df[col].dtype)} 
                    for col in df.columns
                ]
                meta['preview'] = df.head(5).to_dict(orient='records')
            
            elif mime_type in (
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-excel'
            ):
                import pandas as pd
                df = pd.read_excel(path, nrows=5)
                
                meta['column_count'] = len(df.columns)
                meta['columns'] = [
                    {'name': col, 'dtype': str(df[col].dtype)} 
                    for col in df.columns
                ]
                meta['preview'] = df.head(5).to_dict(orient='records')
        
        except Exception as e:
            logger.warning(f"Could not analyze file {path}: {e}")
        
        return meta


# Singleton
file_manager = FileManager()
