"""
fetch_from_url.py
-----------------
MCP Tool: fetch_from_url

Allows the sandbox to pull a file directly from any accessible URL
(Cloudinary CDN, S3, public HTTPS) and save it into the sandbox data
directory so execute_code can work with it.

Uses stdlib urllib (no httpx dependency) for maximum reliability.
"""

import os
import re
import logging
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# -- Config ----------------------------------------------------------------
SANDBOX_BASE = Path(os.environ.get("SANDBOX_DATA_DIR", "/app/sandbox_data"))
MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FETCH_SIZE_MB", "500")) * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".xlsx", ".xls", ".csv", ".tsv",
    ".json", ".jsonl", ".parquet",
    ".pdf", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".zip", ".tar", ".gz",
    ".db", ".sqlite",
}
TIMEOUT_SECONDS = 60


def _infer_filename(url: str, content_disposition: str | None = None) -> str:
    """Extract a safe filename from URL or Content-Disposition header."""
    # Try Content-Disposition first
    if content_disposition:
        match = re.search(r'filename[^;=\n]*=(["\']?)([^"\'\n;]+)\1', content_disposition)
        if match:
            return match.group(2).strip()

    # Fall back to URL path
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name and "." in name:
        return name

    return "downloaded_file"


def _sanitize_filename(filename: str) -> str:
    """Remove path traversal and dangerous characters."""
    # Strip directory components
    name = Path(filename).name
    # Replace anything that isn't alphanumeric, dash, underscore, dot
    name = re.sub(r"[^\w\-.]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name or "file"


def fetch_from_url(
    url: str,
    filename: str | None = None,
    session_id: str = "default",
) -> dict:
    """
    Fetch a file from a URL and save it to the sandbox.

    Args:
        url:        Full HTTPS URL to the file.
        filename:   Optional filename to save as. Inferred from URL if omitted.
        session_id: Sandbox session directory. Defaults to "default".

    Returns:
        dict with success, path, size_bytes, filename — or error message.
    """
    logger.info(f"fetch_from_url: url={url!r}, filename={filename!r}, session={session_id!r}")

    # -- Validate URL ----------------------------------------------------------
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"success": False, "error": f"Only http/https URLs are supported. Got: {parsed.scheme!r}"}

    # -- Prepare destination ---------------------------------------------------
    session_dir = SANDBOX_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # -- Fetch -----------------------------------------------------------------
    try:
        req = Request(url, headers={"User-Agent": "PowerInterpreter/1.0"})
        with urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            # Infer filename if not provided
            content_disp = response.headers.get("Content-Disposition")
            if not filename:
                filename = _infer_filename(url, content_disp)
            filename = _sanitize_filename(filename)

            # Check extension
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return {
                    "success": False,
                    "error": f"File extension {ext!r} not allowed. Permitted: {sorted(ALLOWED_EXTENSIONS)}",
                }

            # Stream to disk with size guard
            dest_path = session_dir / filename
            total_bytes = 0
            chunk_size = 65536  # 64KB chunks

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE_BYTES:
                        dest_path.unlink(missing_ok=True)
                        return {
                            "success": False,
                            "error": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                        }
                    f.write(chunk)

    except HTTPError as e:
        logger.error(f"fetch_from_url HTTP error: {e.code} {e.reason} for {url}")
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}"}
    except URLError as e:
        logger.error(f"fetch_from_url URL error: {e.reason} for {url}")
        return {"success": False, "error": f"URL error: {e.reason}"}
    except Exception as e:
        logger.exception(f"fetch_from_url unexpected error for {url}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

    logger.info(f"fetch_from_url: saved {filename} ({total_bytes:,} bytes) -> {dest_path}")

    return {
        "success": True,
        "path": str(dest_path),
        "filename": filename,
        "size_bytes": total_bytes,
        "session_id": session_id,
        "message": f"File saved to sandbox at {dest_path}. You can now open it with openpyxl.load_workbook('{dest_path}') or pd.read_excel('{dest_path}').",
    }
