"""Power Interpreter - Start Script

Reads the PORT environment variable (set by Railway) and starts uvicorn.
Uses single worker to avoid multiprocessing issues on Railway.
"""

import os
import sys
import uvicorn

# Ensure project root is on path for app.version import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.version import __version__
except ImportError:
    __version__ = "2.9.1"  # fallback if import fails

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    has_db = bool(os.environ.get("DATABASE_URL"))
    has_api_key = bool(os.environ.get("API_KEY"))
    public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    has_ms_config = bool(
        os.environ.get("MICROSOFT_CLIENT_ID")
        and os.environ.get("MICROSOFT_CLIENT_SECRET")
        and os.environ.get("MICROSOFT_TENANT_ID")
    )

    print("=" * 60)
    print(f"Power Interpreter MCP v{__version__} starting on port {port}")
    print("=" * 60)
    print(f"  DATABASE_URL:  {'configured' if has_db else 'NOT SET (will start without DB)'}")
    print(f"  API_KEY:       {'configured' if has_api_key else 'NOT SET (dev mode)'}")
    print(f"  PUBLIC_DOMAIN: {public_domain or 'NOT SET'}")
    print(f"  Microsoft:     {'ENABLED' if has_ms_config else 'not configured (optional)'}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        access_log=True,
    )
