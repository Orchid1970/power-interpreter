"""Power Interpreter - File Management Routes

Handles file upload (base64), file fetch (URL), file listing,
and public download of sandbox-generated files from Postgres.

Version: 1.4.0 - Fix: Convert OneDrive/SharePoint sharing URLs to direct
                 download URLs via the OneDrive API shares endpoint.
                 
                 URL format: /dl/{file_id}/{filename}
                 Fallback:   /dl/{file_id} (still works, uses DB filename)
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import base64
import hashlib
import mimetypes
import time
import logging
import httpx
import uuid
from urllib.parse import quote

from app.config import settings

logger = logging.Logger(__name__)

router = APIRouter()

# Separate router for public download (no API key required)
# This s mounted at /dl in main.py
public_router = APIRouter()

SANDBOX_DIR = settings.SANDBOX_DIR

# Max file sizes
MAX_UPLOAD_SIZE = 50 * 1024 * 1024   # 50MB for base64 uploads
MAX_FETCH_SIZE = 500 * 1024 * 1024   # 500MB for URL fetches


# ============================================================
# CORS headers for public download endpoint
# ============================================================
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": ", HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Content-Type, X-Session-Id",
    "Access-Control-Max-Age": "3600",
}


# ============================================================
# Cloud URL Converters
# ============================================================

_ONEDRIVE_PATTERNS = [
    '1drv.ms/',
    'onedrive.live.com/',
    '.sharepoint.com/:',
    'onedrive.live.com/redir',
]


def _convert_onedrive_to_direct(url: str) -> str:
    """Convert OneDrive/SharePoint sharing URLs to direct download URLs.

    OneDrive sharing links (1drv.ms, onedrive.live.com) redirect to an HTML
    viewer page, not the raw file. This converts them to direct download URLs
    using the OneDrive API shares endpoint.

    Supports:
    - https://1drv.ms/x/... (short links)
    - https://onedrive.live.com/... (full links)
    - https://*.sharepoint.com/:x:/... (SharePoint links)

    Returns the original URL unchanged if it's not a OneDrive/SharePoint link.
    """
    is_onedrive = any(pattern in url for pattern in _ONEDRIVE_PATTERNS)
    if not is_onedrive:
        return url

    # OneDrive API shares endpoint trick:
    # 1. Base64-encode the sharing URL
    # 2. Replace / with _, + with -, strip trailing =
    # 3. Prefix with "u!"
    # 4. Hit https://api.onedrive.com/v1.0/shares/{token}/root/content
    encoded = base64.b64encode(url.encode('utf-8')).decode('utf-8')
    encoded = encoded.rstrip('=').replace('/', '_').replace('+', '-')
    share_token = f"u!{encoded}"

    direct_url = f"https://api.onedrive.com/v1.0/shares/{share_token}/root/content"
    logger.info(f"OneDrive URL converted: {url[:80]}... -> API shares endpoint")
    return direct_url


# ============================================================
# Request/Response Models
# ============================================================

class UploadFileRequest(BaseModel):
    """Upload a file via base64 content"""
    filename: str = Field(..., description="Filename (e.g., 'data.csv')")
    content_base64: str = Field(..., description="Base64-encoded file content")
    session_id: str = Field(default="default", description="Session for isolation")


class FetchFileRequest(BaseModel):
    """Fetch a file from a URL"""
    url: str = Field(..., description="URL to download from")
    filename: str = Field(..., description="Name to save as (e.g., 'invoices.csv')")
    session_id: str = Field(default="default", description="Session for isolation")


class FileInfo(BaseModel):
    """Information about a file in the sandbox"""
    filename: str
    path: str
    size_bytes: int
    size_human: str
    mime_type: Optional[str] = None
    session_id: str
    preview: Optional[str] = None


class FileListResponse(BaseModel):
    """Response for file listing"""
    files: List[FileInfo]
    total_count: int
    total_size_bytes: int
    total_size_human: str


# ============================================================
# Helper Functions
# ============================================================

def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _safe_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    name = Path(filename).name
    name = name.replace('..', '_').replace('/', '_').replace('\\', '_')
    if not name or name.startswith('.'):
        name = f"file_{int(time.time())}"
    return name


def __preview(file_path: Path, max_lines: int = 5) -> Optional[str]:
    """Get a text preview of a file"""
    try:
        suffix = file_path.suffix.lower()
        if suffix in ['.csv', '.tsv', '.txt', '.json', '.md', '.log']:
            with open(file_path, 'r', errors='replace') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
                return '\n'.join(lines)
        elif suffix in ['.xlsx', '.xls']:
            return f"[Excel file - use execute_code with pandas to read]"
        elif suffix == '.pdf':
            return f"[PDF file - use execute_code with pdfplumber to read]"
        else:
            return f"[Binary file: {suffix}]"
    except Exception:
        return None


def _detect_mime_type(filename: str) -> Optional[str]:
    """Detect MIME type from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type


# ============================================================
# MIME type mapping for common generated files
# ============================================================

MIME_TYPES = {
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv',
    '.tsv': 'text/tab-separated-values',
    '.json': 'application/json',
    '.pdf': 'application/pdf',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.html': 'text/html',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.zip': 'application/zip',
    '.parquet': 'application/octet-stream',
}


def get_mime_type(filename: str) -> str:
    """Get MIME type for a filename, with good defaults for common types."""
    ext = Path(filename).suffix.lower()
    return MIME_TYPES.get(ext, 'application/octet-stream')


# ============================================================
# Core download logic (shared by all download routes)
# ============================================================

async def _serve_download(file_id: str, head_only: bool = False) -> Response:
    """Core logic to serve a file download from Postgres."""
    try:
        file_uuid = uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID format")

    try:
        from app.database import get_session_factory
        from app.models import SandboxFile
        from sqlalchemy import select

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(SandboxFile).where(SandboxFile.id == file_uuid)
            )
            sandbox_file = result.scalar_one_or_none()

            if not sandbox_file:
                raise HTTPException(status_code=404, detail="File not found or expired")

            if sandbox_file.expires_at:
                from datetime import datetime
                if datetime.utcnow() > sandbox_file.expires_at:
                    raise HTTPException(status_code=410, detail="File has expired")

            if not head_only:
                sandbox_file.download_count = (sandbox_file.download_count or 0) + 1
                await session.commit()

            safe_name = sandbox_file.filename
            encoded_name = quote(safe_name)
            ascii_name = safe_name.encode('ascii', 'replace').decode('ascii')
            mime = sandbox_file.mime_type or 'application/octet-stream'
            IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml', 'image/gif', 'image/webp'}
      if mime in IMAGE_MIME_TYPES:
        disposition = (
          f'inline; '
          f'filename="{ascii_name}"; '
          f'filename*=UTF-8\'\'{encoded_name}'
        )
      else:
        disposition = (
          f'attachment; '
          f'filename="{ascii_name}"; '
          f'filename*=UTF-8\'\'{encoded_name}'
        )


            logger.info(
                f"download: {sandbox_file.filename} "
                f"({_human_size(sandbox_file.file_size)}) "
                f"session={sandbox_file.session_id} "
                f"downloads={sandbox_file.download_count}"
            )

            headers = {
                "Content-Disposition": disposition,
                "Content-Length": str(sandbox_file.file_size),
                "Cache-Control": "private, max-age=3600",
                "X-Session-Id": sandbox_file.session_id,
                **CORS_HEADERS,
            }

            if head_only:
                return Response(
                    status_code=200,
                    media_type=mime,
                    headers=headers,
                )

            return Response(
                content=sandbox_file.content,
                media_type=mime,
                headers=headers,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"download error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file")


# ============================================================
# Public Download Endpoints (NO API key required)
# ============================================================

@public_router.options("/{file_id}")
async def download_preflight(file_id: str):
    return Response(status_code=204, headers=CORS_HEADERS)


@public_router.options("/{file_id}/{filename}")
async def download_preflight_with_name(file_id: str, filename: str):
    return Response(status_code=204, headers=CORS_HEADERS)


@public_router.head("/{file_id}")
async def download_head(file_id: str):
    return await _serve_download(file_id, head_only=True)


@public_router.head("/{file_id}/{filename}")
async def download_head_with_name(file_id: str, filename: str):
    return await _serve_download(file_id, head_only=True)


@public_router.get("/{file_id}/{filename}")
async def download_with_filename(file_id: str, filename: str):
    """Download with filename in URL path (preferred format)."""
    return await _serve_download(file_id)


@public_router.get("/{file_id}")
async def download_sandbox_file(file_id: str):
    """Download with file_id only (backward compatible)."""
    return await _serve_download(file_id)


# ============================================================
# Upload Endpoint (Base64)
# ============================================================

@router.post("/files/upload", response_model=FileInfo)
async def upload_file(request: UploadFileRequest):
    """Upload a file to the sandbox via base64 encoding."""
    logger.info(f"upload_file: filename={request.filename}, session={request.session_id}")

    safe_name = _safe_filename(request.filename)

    try:
        file_bytes = base64.b64decode(request.content_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {e}")

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {_human_size(len(file_bytes))}. "
                   f"Max upload size is {_human_size(MAX_UPLOAD_SIZE)}. "
                   f"Use fetch_file with a URL for larger files."
        )

    session_dir = SANDBOX_DIR / request.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / safe_name
    file_path.write_bytes(file_bytes)

    logger.info(f"upload_file: saved {safe_name} ({len(file_bytes)} bytes) "
                f"to {file_path}")

    preview = _get_preview(file_path)

    return FileInfo(
        filename=safe_name,
        path=str(file_path.relative_to(SANDBOX_DIR)),
        size_bytes=len(file_bytes),
        size_human=_human_size(len(file_bytes)),
        mime_type=_detect_mime_type(safe_name),
        session_id=request.session_id,
        preview=preview
    )


# ============================================================
# Fetch Endpoint (URL Download)
# ============================================================

@router.post("/files/fetch", response_model=FileInfo)
async def fetch_file(request: FetchFileRequest):
    """Download a file from a URL into the sandbox.

    Supports:
    - Direct download URLs
    - OneDrive / SharePoint sharing links (auto-converted to direct download)
    - Google Drive sharing links (use export format)
    - Dropbox links (change dl=0 to dl=1)
    - S3 pre-signed URLs
    - Any publicly accessible HTTP/HTTPS URL
    """
    logger.info(f"fetch_file: url={request.url[:100]}..., "
                f"filename={request.filename}, session={request.session_id}")

    safe_name = _safe_filename(request.filename)

    if not request.url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Convert cloud sharing URLs to direct download URLs
    download_url = _convert_onedrive_to_direct(request.url)

    session_dir = SANDBOX_DIR / request.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / safe_name

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30),
            follow_redirects=True,
            max_redirects=10
        ) as client:
            async with client.stream("GET", download_url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to download: HTTP {response.status_code} from {request.url[:100]}"
                    )

                # Validate content-type: reject HTML responses for binary file downloads
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type and safe_name.lower().endswith(
                    ('.xlsx', '.xls', '.csv', '.pdf', '.zip', '.parquet', '.docx', '.doc')
                ):
                    raise HTTPException(
                        status_code=502,
                        detail=f"URL returned HTML instead of the expected file. "
                               f"Content-Type: {content_type}. "
                               f"The sharing link may require authentication or may not support direct download."
                    )

                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_FETCH_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {_human_size(int(content_length))}. "
                               f"Max fetch size is {_human_size(MAX_FETCH_SIZE)}."
                    )

                total_bytes = 0
                with open(file_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_FETCH_SIZE:
                            f.close()
                            file_path.unlink(missing_ok=True)
                            raise HTTPException(
                                status_code=413,
                                detail=f"File exceeded max size during download. "
                                       f"Max fetch size is {_human_size(MAX_FETCH_SIZE)}."
                            )
                        f.write(chunk)

    except httpx.TimeoutException:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=504,
            detail=f"Download timed out after 300 seconds."
        )
    except httpx.RequestError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download from URL: {e}"
        )
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during download: {e}"
        )

    file_size = file_path.stat().st_size
    logger.info(f"fetch_file: downloaded {safe_name} ({file_size} bytes) "
                f"from {request.url[:80]}")

    preview = _get_preview(file_path)

    return FileInfo(
        filename=safe_name,
        path=str(file_path.relative_to(SANDBOX_DIR)),
        size_bytes=file_size,
        size_human=_human_size(file_size),
        mime_type=_detect_mime_type(safe_name),
        session_id=request.session_id,
        preview=preview
    )


# ============================================================
# List Files Endpoint
# ============================================================

@router.get("/files", response_model=FileListResponse)
async def list_files(session_id: Optional[str] = None):
    """List files in the sandbox."""
    files = []
    total_size = 0

    if session_id:
        session_dir = SANDBOX_DIR / session_id
        if session_dir.exists():
            for file_path in sorted(session_dir.rglob('*')):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    files.append(FileInfo(
                        filename=file_path.name,
                        path=str(file_path.relative_to(SANDBOX_DIR)),
                        size_bytes=size,
                        size_human=_human_size(size),
                        mime_type=_detect_mime_type(file_path.name),
                        session_id=session_id,
                        preview=_get_preview(file_path)
                    ))
    else:
        if SANDBOX_DIR.exists():
            for session_dir in sorted(SANDBOX_DIR.iterdir()):
                if session_dir.is_dir():
                    sid = session_dir.name
                    for file_path in sorted(session_dir.rglob('*')):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            total_size += size
                            files.append(FileInfo(
                                filename=file_path.name,
                                path=str(file_path.relative_to(SANDBOX_DIR)),
                                size_bytes=size,
                                size_human=_human_size(size),
                                mime_type=_detect_mime_type(file_path.name),
                                session_id=sid,
                                preview=_get_preview(file_path)
                            ))

    return FileListResponse(
        files=files,
        total_count=len(files),
        total_size_bytes=total_size,
        total_size_human=_human_size(total_size)
    )


# ============================================================
# Delete File Endpoint
# ============================================================

@router.delete("/files/{session_id}/{filename}")
async def delete_file(session_id: str, filename: str):
    """Delete a file from the sandbox."""
    safe_name = _safe_filename(filename)
    file_path = SANDBOX_DIR / session_id / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")

    try:
        file_path.resolve().relative_to(SANDBOX_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    file_path.unlink()
    logger.info(f"delete_file: deleted {session_id}/{safe_name}")

    return {"deleted": safe_name, "session_id": session_id}
