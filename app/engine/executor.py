"""Power Interpreter - Sandboxed Code Executor

Executes Python code in a controlled environment with:
- Resource limits (time, memory)
- Restricted imports (whitelist only)
- Safe file I/O (sandbox directory only)
- Captured stdout/stderr
- Structured result extraction
- PERSISTENT SESSION STATE (v2.0) - variables survive across calls
- AUTO FILE STORAGE (v2.1) - generated files stored in Postgres with download URLs
- INLINE CHART RENDERING (v2.4) - matplotlib/plotly charts auto-captured and
  returned as inline image URLs for rendering directly in chat

Design: Uses in-process isolation with restricted globals.
The sandbox has access to pandas, numpy, matplotlib, etc.
but cannot access the network, filesystem outside sandbox, or system commands.

Session state is managed by KernelManager - globals dicts are persisted
per session_id and reused across execute() calls, giving notebook-like
continuity.

Generated files (xlsx, png, csv, etc.) are automatically stored in
Postgres (sandbox_files table) and download URLs are returned so they
survive Railway container redeployments.

Chart auto-capture: plt.show() is monkey-patched to save the current
figure as PNG, store it in Postgres, and return the URL. This means
any matplotlib code that calls plt.show() will produce an inline image.

Version: 2.4.0 - Inline chart rendering via plt.show() auto-capture
                 Plotly figures also auto-captured via pio.write_image fallback
"""

import asyncio
import io
import os
import sys
import time
import traceback
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from contextlib import redirect_stdout, redirect_stderr
import logging
import uuid
from urllib.parse import quote

from app.config import settings
from app.engine.kernel_manager import kernel_manager

logger = logging.getLogger(__name__)


# Try to import resource module (Unix only, may fail in some containers)
try:
    import resource as resource_module
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    logger.warning("resource module not available - memory limits disabled")


# ============================================================
# File extensions we auto-store in Postgres for download URLs
# ============================================================
STORABLE_EXTENSIONS = {
    '.xlsx', '.xls', '.csv', '.tsv', '.json',
    '.pdf', '.png', '.jpg', '.jpeg', '.svg',
    '.html', '.txt', '.md', '.zip', '.parquet',
}

# Image extensions for inline rendering
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg'}


class ExecutionResult:
    """Result of code execution"""
    def __init__(self):
        self.success: bool = False
        self.stdout: str = ""
        self.stderr: str = ""
        self.result: Any = None
        self.error_message: Optional[str] = None
        self.error_traceback: Optional[str] = None
        self.execution_time_ms: int = 0
        self.memory_used_mb: float = 0.0
        self.files_created: list = []
        self.download_urls: list = []  # Public download URLs
        self.inline_images: list = []  # Image URLs for inline rendering
        self.variables: Dict[str, str] = {}  # Variable name -> type string
        self.kernel_info: Dict[str, Any] = {}  # Kernel session metadata

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'stdout': self.stdout[:settings.MAX_OUTPUT_SIZE],
            'stderr': self.stderr[:settings.MAX_OUTPUT_SIZE],
            'result': self._serialize_result(),
            'error_message': self.error_message,
            'error_traceback': self.error_traceback,
            'execution_time_ms': self.execution_time_ms,
            'memory_used_mb': round(self.memory_used_mb, 2),
            'files_created': self.files_created,
            'download_urls': self.download_urls,
            'inline_images': self.inline_images,
            'variables': self.variables,
            'kernel_info': self.kernel_info,
        }

    def _serialize_result(self) -> Any:
        """Safely serialize the result"""
        if self.result is None:
            return None
        try:
            import json
            json.dumps(self.result)
            return self.result
        except (TypeError, ValueError):
            return str(self.result)[:settings.MAX_OUTPUT_SIZE]


class ChartCapture:
    """Captures matplotlib figures when plt.show() is called.
    
    This replaces plt.show() with a function that:
    1. Saves the current figure as a high-quality PNG
    2. Adds the path to a capture list
    3. Closes the figure to free memory
    
    The executor then stores these PNGs in Postgres and returns
    inline image URLs.
    """
    
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.captured_charts: List[str] = []  # List of file paths
        self._chart_counter = 0
    
    def make_show_replacement(self, plt_module):
        """Create a replacement for plt.show() that captures figures."""
        capture = self  # closure reference
        
        def _capturing_show(*args, **kwargs):
            """Replacement for plt.show() that saves figures as PNG."""
            try:
                # Get all open figures
                fig_nums = plt_module.get_fignums()
                if not fig_nums:
                    logger.debug("plt.show() called but no figures to capture")
                    return
                
                for fig_num in fig_nums:
                    fig = plt_module.figure(fig_num)
                    capture._chart_counter += 1
                    
                    # Generate filename
                    chart_name = f"chart_{capture._chart_counter:03d}.png"
                    chart_path = capture.session_dir / chart_name
                    
                    # Save high-quality PNG
                    fig.savefig(
                        str(chart_path),
                        format='png',
                        dpi=150,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1,
                    )
                    
                    capture.captured_charts.append(chart_name)
                    logger.info(f"Chart captured: {chart_name} (figure {fig_num})")
                
                # Close all figures to free memory
                plt_module.close('all')
                
            except Exception as e:
                logger.error(f"Chart capture failed: {e}", exc_info=True)
                # Don't raise - let the code continue even if capture fails
        
        return _capturing_show
    
    def make_savefig_wrapper(self, original_savefig, plt_module):
        """Wrap plt.savefig() to also track saved figures.
        
        Users who call plt.savefig() explicitly still get their file,
        but we also track it for inline rendering.
        """
        capture = self
        
        def _tracking_savefig(self_fig, fname, *args, **kwargs):
            """Wrapper around Figure.savefig that tracks output files."""
            # Call original savefig
            result = original_savefig(self_fig, fname, *args, **kwargs)
            
            # Track the file if it's an image
            try:
                fname_path = Path(fname)
                if fname_path.suffix.lower() in IMAGE_EXTENSIONS:
                    # If relative path, it's in session_dir (because we chdir)
                    if not fname_path.is_absolute():
                        capture.captured_charts.append(str(fname_path))
                        logger.info(f"Tracked savefig: {fname}")
            except Exception as e:
                logger.debug(f"Could not track savefig: {e}")
            
            return result
        
        return _tracking_savefig


class SandboxExecutor:
    """Executes Python code in a sandboxed environment with persistent state"""

    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or settings.SANDBOX_DIR
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SandboxExecutor initialized: sandbox_dir={self.sandbox_dir}")

    def _get_safe_builtins(self) -> Dict:
        """Build a safe builtins dict, handling both dict and module forms."""
        safe = {}

        # Get the actual builtins dict regardless of form
        import builtins as builtins_module

        blocked = set(settings.BLOCKED_BUILTINS) if hasattr(settings, 'BLOCKED_BUILTINS') else {
            'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
            'exit', 'quit', 'breakpoint', 'input',
        }

        for name in dir(builtins_module):
            if name.startswith('_'):
                continue
            if name in blocked:
                continue
            try:
                safe[name] = getattr(builtins_module, name)
            except AttributeError:
                pass

        # Always include these essentials
        safe['True'] = True
        safe['False'] = False
        safe['None'] = None
        safe['print'] = print
        safe['len'] = len
        safe['range'] = range
        safe['int'] = int
        safe['float'] = float
        safe['str'] = str
        safe['bool'] = bool
        safe['list'] = list
        safe['dict'] = dict
        safe['tuple'] = tuple
        safe['set'] = set
        safe['type'] = type
        safe['isinstance'] = isinstance
        safe['issubclass'] = issubclass
        safe['hasattr'] = hasattr
        safe['getattr'] = getattr
        safe['setattr'] = setattr
        safe['enumerate'] = enumerate
        safe['zip'] = zip
        safe['map'] = map
        safe['filter'] = filter
        safe['sorted'] = sorted
        safe['reversed'] = reversed
        safe['min'] = min
        safe['max'] = max
        safe['sum'] = sum
        safe['abs'] = abs
        safe['round'] = round
        safe['any'] = any
        safe['all'] = all
        safe['repr'] = repr
        safe['format'] = format
        safe['chr'] = chr
        safe['ord'] = ord
        safe['hex'] = hex
        safe['oct'] = oct
        safe['bin'] = bin
        safe['pow'] = pow
        safe['divmod'] = divmod
        safe['hash'] = hash
        safe['id'] = id
        safe['dir'] = dir
        safe['vars'] = vars
        safe['iter'] = iter
        safe['next'] = next
        safe['slice'] = slice
        safe['super'] = super
        safe['property'] = property
        safe['staticmethod'] = staticmethod
        safe['classmethod'] = classmethod
        safe['object'] = object
        safe['Exception'] = Exception
        safe['ValueError'] = ValueError
        safe['TypeError'] = TypeError
        safe['KeyError'] = KeyError
        safe['IndexError'] = IndexError
        safe['AttributeError'] = AttributeError
        safe['RuntimeError'] = RuntimeError
        safe['StopIteration'] = StopIteration
        safe['ZeroDivisionError'] = ZeroDivisionError
        safe['FileNotFoundError'] = FileNotFoundError
        safe['IOError'] = IOError
        safe['OSError'] = OSError
        safe['PermissionError'] = PermissionError
        safe['NotImplementedError'] = NotImplementedError
        safe['OverflowError'] = OverflowError

        return safe

    def _build_safe_globals(self, session_dir: Path) -> Dict:
        """Build the globals dict for sandboxed execution"""
        logger.info("Building safe globals...")

        try:
            import pandas as pd
        except ImportError:
            pd = None
            logger.warning("pandas not available")

        try:
            import numpy as np
        except ImportError:
            np = None
            logger.warning("numpy not available")

        import json
        import csv
        import math
        import statistics
        import datetime as dt_module
        import collections
        import itertools
        import functools
        import re
        import io as io_module
        import copy
        import hashlib as hashlib_module
        import base64
        from decimal import Decimal
        from fractions import Fraction
        from pathlib import Path as PathLib

        # Try dataclasses
        try:
            from dataclasses import dataclass, field, asdict
        except ImportError:
            dataclass = None
            field = None
            asdict = None

        from typing import Dict, List, Optional, Tuple, Set, Any

        # Get safe builtins
        safe_builtins = self._get_safe_builtins()

        # Add safe file operations
        safe_builtins['open'] = self._make_safe_open(session_dir)

        # Build globals
        sandbox_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',

            # Data libraries
            'json': json,
            'csv': csv,

            # Math & stats
            'math': math,
            'statistics': statistics,
            'Decimal': Decimal,
            'Fraction': Fraction,

            # Standard library
            'datetime': dt_module,
            'collections': collections,
            'itertools': itertools,
            'functools': functools,
            're': re,
            'io': io_module,
            'copy': copy,
            'hashlib': hashlib_module,
            'base64': base64,
            'Path': PathLib,

            # Typing
            'Dict': Dict,
            'List': List,
            'Optional': Optional,
            'Tuple': Tuple,
            'Set': Set,
            'Any': Any,

            # Sandbox info
            'SANDBOX_DIR': str(session_dir),
            'RESULT': None,  # User can set this for structured output
        }

        # Add pandas/numpy if available
        if pd is not None:
            sandbox_globals['pd'] = pd
            sandbox_globals['pandas'] = pd
        if np is not None:
            sandbox_globals['np'] = np
            sandbox_globals['numpy'] = np

        # Add dataclasses if available
        if dataclass is not None:
            sandbox_globals['dataclass'] = dataclass
            sandbox_globals['field'] = field
            sandbox_globals['asdict'] = asdict

        logger.info("Safe globals built successfully")
        return sandbox_globals

    def _make_safe_open(self, session_dir: Path):
        """Create a safe open() that only allows access within session directory"""
        def safe_open(filepath, mode='r', *args, **kwargs):
            # Resolve the path
            path = Path(filepath)
            if not path.is_absolute():
                path = session_dir / path

            # Ensure it's within the sandbox
            try:
                resolved = path.resolve()
                session_resolved = session_dir.resolve()
                if not str(resolved).startswith(str(session_resolved)):
                    raise PermissionError(
                        f"Access denied: Cannot access files outside sandbox. "
                        f"Use relative paths or SANDBOX_DIR."
                    )
            except PermissionError:
                raise
            except Exception as e:
                raise PermissionError(f"Invalid file path: {e}")

            # Create parent directories if writing
            if 'w' in mode or 'a' in mode:
                resolved.parent.mkdir(parents=True, exist_ok=True)

            return open(resolved, mode, *args, **kwargs)

        return safe_open

    def _install_chart_hooks(self, sandbox_globals: Dict, chart_capture: ChartCapture):
        """Install matplotlib hooks for auto-capturing charts.
        
        This replaces plt.show() with our capturing version and wraps
        plt.savefig() to track explicitly saved images.
        
        Called AFTER _lazy_import loads matplotlib, and EVERY execution
        (because the user might have reassigned plt in their code).
        """
        plt = sandbox_globals.get('plt')
        if plt is None:
            return
        
        # Replace plt.show() with our capturing version
        plt.show = chart_capture.make_show_replacement(plt)
        
        # Wrap Figure.savefig to track explicit saves
        try:
            import matplotlib.figure
            if not hasattr(matplotlib.figure.Figure, '_original_savefig'):
                matplotlib.figure.Figure._original_savefig = matplotlib.figure.Figure.savefig
            matplotlib.figure.Figure.savefig = chart_capture.make_savefig_wrapper(
                matplotlib.figure.Figure._original_savefig, plt
            )
        except Exception as e:
            logger.debug(f"Could not wrap Figure.savefig: {e}")
        
        logger.debug("Chart capture hooks installed")

    def _lazy_import(self, name: str, sandbox_globals: Dict):
        """Lazily import allowed libraries into sandbox.

        Returns True if the module was loaded (or already available).
        When loading, also imports common submodules so that
        'from X.Y import Z' can be resolved via attribute access.
        """
        try:
            if name == 'matplotlib' or name == 'matplotlib.pyplot':
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                import matplotlib.patches
                import matplotlib.gridspec
                import matplotlib.ticker
                import matplotlib.colors
                import matplotlib.cm
                try:
                    import matplotlib.patheffects
                except ImportError:
                    pass
                try:
                    import matplotlib.image
                except ImportError:
                    pass
                sandbox_globals['matplotlib'] = matplotlib
                sandbox_globals['plt'] = plt
                # NOTE: Chart hooks are installed in execute() after preprocessing
                return True
            elif name == 'seaborn':
                import seaborn as sns
                sandbox_globals['sns'] = sns
                sandbox_globals['seaborn'] = sns
                return True
            elif name == 'plotly' or name == 'plotly.express':
                import plotly
                import plotly.express as px
                import plotly.graph_objects as go
                sandbox_globals['plotly'] = plotly
                sandbox_globals['px'] = px
                sandbox_globals['go'] = go
                return True
            elif name == 'scipy' or name == 'scipy.stats':
                import scipy
                import scipy.stats
                try:
                    import scipy.optimize
                except ImportError:
                    pass
                try:
                    import scipy.interpolate
                except ImportError:
                    pass
                sandbox_globals['scipy'] = scipy
                return True
            elif name == 'sklearn':
                import sklearn
                sandbox_globals['sklearn'] = sklearn
                return True
            elif name == 'statsmodels':
                import statsmodels
                import statsmodels.api as sm
                sandbox_globals['statsmodels'] = statsmodels
                sandbox_globals['sm'] = sm
                return True
            elif name == 'openpyxl':
                import openpyxl
                import openpyxl.styles
                import openpyxl.utils
                try:
                    import openpyxl.chart
                except ImportError:
                    pass
                try:
                    import openpyxl.worksheet.table
                except ImportError:
                    pass
                try:
                    import openpyxl.formatting
                    import openpyxl.formatting.rule
                except ImportError:
                    pass
                sandbox_globals['openpyxl'] = openpyxl
                return True
            elif name == 'xlsxwriter':
                import xlsxwriter
                sandbox_globals['xlsxwriter'] = xlsxwriter
                return True
            elif name == 'pdfplumber':
                import pdfplumber
                sandbox_globals['pdfplumber'] = pdfplumber
                return True
            elif name == 'tabulate':
                from tabulate import tabulate
                sandbox_globals['tabulate'] = tabulate
                return True
            elif name == 'textwrap':
                import textwrap
                sandbox_globals['textwrap'] = textwrap
                return True
            elif name == 'string':
                import string
                sandbox_globals['string'] = string
                return True
            elif name == 'struct':
                import struct
                sandbox_globals['struct'] = struct
                return True
            elif name == 'decimal':
                import decimal
                sandbox_globals['decimal'] = decimal
                return True
            elif name == 'fractions':
                import fractions
                sandbox_globals['fractions'] = fractions
                return True
            elif name == 'random':
                import random
                sandbox_globals['random'] = random
                return True
            elif name == 'time':
                import time as time_module
                sandbox_globals['time'] = time_module
                return True
            elif name == 'calendar':
                import calendar
                sandbox_globals['calendar'] = calendar
                return True
            elif name == 'pprint':
                import pprint
                sandbox_globals['pprint'] = pprint
                return True
            elif name == 'dataclasses':
                import dataclasses
                sandbox_globals['dataclasses'] = dataclasses
                return True
            elif name == 'typing':
                import typing
                sandbox_globals['typing'] = typing
                return True
            elif name == 'pathlib':
                import pathlib
                sandbox_globals['pathlib'] = pathlib
                return True
            elif name == 'os':
                # Provide limited os functionality
                import os as os_module
                sandbox_globals['os'] = os_module
                return True
            elif name == 'urllib':
                # Allow urllib for HTTP operations within sandbox
                import urllib
                import urllib.request
                import urllib.parse
                import urllib.error
                sandbox_globals['urllib'] = urllib
                return True
            elif name == 'requests':
                # Allow requests if installed
                try:
                    import requests as requests_module
                    sandbox_globals['requests'] = requests_module
                    return True
                except ImportError:
                    logger.warning("requests not installed")
                    return False
        except ImportError as e:
            logger.warning(f"Failed to import {name}: {e}")
            return False
        return False

    def _preprocess_code(self, code: str, sandbox_globals: Dict) -> str:
        """Preprocess code to handle imports and add safety wrappers.

        Key behavior for 'from X.Y import A, B, C':
        - If base module X is loaded (via _lazy_import), we convert the
          from-import into direct attribute assignments:
            A = X.Y.A
            B = X.Y.B
            C = X.Y.C
        - This works because _lazy_import ensures submodules are imported,
          so attribute access resolves correctly.
        - This avoids needing __import__ (which is blocked in sandbox).
        """
        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                processed_lines.append(line)
                continue

            # Handle import statements
            if stripped.startswith('import ') or stripped.startswith('from '):

                # ============================================================
                # Case 1: 'from X.Y import A, B, C' or 'from X import A, B'
                # ============================================================
                if stripped.startswith('from '):
                    try:
                        # Parse: from <module_path> import <names>
                        parts = stripped.split(' import ', 1)
                        if len(parts) == 2:
                            module_path = parts[0].replace('from ', '').strip()
                            import_names_str = parts[1].strip()
                            base_module = module_path.split('.')[0]

                            # Try to lazy-load the base module
                            if base_module not in sandbox_globals:
                                self._lazy_import(base_module, sandbox_globals)

                            if base_module in sandbox_globals:
                                # Parse the imported names (handle 'A, B, C' and 'A as X')
                                import_items = [n.strip() for n in import_names_str.split(',')]
                                assignment_lines = []

                                for item in import_items:
                                    item = item.strip()
                                    if not item:
                                        continue

                                    # Handle 'Name as Alias'
                                    if ' as ' in item:
                                        original, alias = item.split(' as ', 1)
                                        original = original.strip()
                                        alias = alias.strip()
                                    else:
                                        original = item
                                        alias = item

                                    # Build assignment: alias = module_path.original
                                    # e.g. Font = openpyxl.styles.Font
                                    assignment_lines.append(
                                        f"{alias} = {module_path}.{original}"
                                    )

                                if assignment_lines:
                                    comment = f"# [sandbox] {stripped} -> resolved via attribute access"
                                    processed_lines.append(comment)
                                    processed_lines.extend(assignment_lines)
                                    logger.debug(f"Resolved from-import: {stripped} -> {assignment_lines}")
                                    continue
                                else:
                                    processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                    continue
                            else:
                                # Base module not available
                                processed_lines.append(
                                    f"# [sandbox] BLOCKED: {stripped} (module {base_module} not in allowed list)"
                                )
                                continue
                        else:
                            # Malformed from-import
                            module = stripped.split()[1].split('.')[0]
                            if self._lazy_import(module, sandbox_globals):
                                processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                continue
                            elif module in sandbox_globals:
                                processed_lines.append(f"# [sandbox] {stripped} -> already available")
                                continue
                            else:
                                processed_lines.append(
                                    f"# [sandbox] BLOCKED: {stripped} (not in allowed list)"
                                )
                                continue
                    except Exception as e:
                        logger.warning(f"Failed to parse from-import '{stripped}': {e}")
                        # Fall through to simple handling
                        try:
                            module = stripped.split()[1].split('.')[0]
                            if self._lazy_import(module, sandbox_globals):
                                processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                continue
                            elif module in sandbox_globals:
                                processed_lines.append(f"# [sandbox] {stripped} -> already available")
                                continue
                        except Exception:
                            pass
                        processed_lines.append(
                            f"# [sandbox] BLOCKED: {stripped} (parse error)"
                        )
                        continue

                # ============================================================
                # Case 2: 'import X' or 'import X as Y' or 'import X.Y'
                # ============================================================
                else:
                    # Extract module name
                    module = stripped.split()[1].split('.')[0].split(',')[0]

                    # Try to lazy-load it
                    if self._lazy_import(module, sandbox_globals):
                        # Handle 'import X as Y'
                        if ' as ' in stripped:
                            alias = stripped.split(' as ')[-1].strip()
                            if module in sandbox_globals:
                                sandbox_globals[alias] = sandbox_globals[module]
                        processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                        continue
                    elif module in sandbox_globals:
                        if ' as ' in stripped:
                            alias = stripped.split(' as ')[-1].strip()
                            sandbox_globals[alias] = sandbox_globals[module]
                        processed_lines.append(f"# [sandbox] {stripped} -> already available")
                        continue
                    else:
                        # Block unknown imports
                        processed_lines.append(
                            f"# [sandbox] BLOCKED: {stripped} (not in allowed list)"
                        )
                        continue

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    # ================================================================
    # File Storage: Store generated files in Postgres for download URLs
    # ================================================================

    async def _store_files_in_postgres(
        self,
        new_file_paths: List[str],
        session_id: str,
        session_dir: Path,
    ) -> List[Dict[str, str]]:
        """Store newly created files in Postgres and return download URL info.

        For each new file:
        1. Read bytes from disk
        2. Check size is under SANDBOX_FILE_MAX_MB
        3. Insert into sandbox_files table
        4. Build public download URL with filename in path

        Returns list of dicts: [{filename, url, size_human, is_image}]
        """
        if not new_file_paths:
            return []

        download_info = []
        max_bytes = settings.SANDBOX_FILE_MAX_MB * 1024 * 1024
        base_url = settings.public_base_url
        ttl_hours = settings.SANDBOX_FILE_TTL_HOURS

        try:
            from app.database import get_session_factory
            from app.models import SandboxFile
            from app.routes.files import get_mime_type

            factory = get_session_factory()
            async with factory() as db_session:
                for rel_path in new_file_paths:
                    try:
                        file_path = session_dir / rel_path
                        if not file_path.exists() or not file_path.is_file():
                            continue

                        # Check extension
                        ext = file_path.suffix.lower()
                        if ext not in STORABLE_EXTENSIONS:
                            logger.debug(f"Skipping {rel_path}: extension {ext} not storable")
                            continue

                        # Check size
                        file_size = file_path.stat().st_size
                        if file_size > max_bytes:
                            logger.warning(
                                f"Skipping {rel_path}: {file_size} bytes exceeds "
                                f"{settings.SANDBOX_FILE_MAX_MB}MB limit"
                            )
                            continue

                        if file_size == 0:
                            logger.debug(f"Skipping {rel_path}: empty file")
                            continue

                        # Read file bytes
                        file_bytes = file_path.read_bytes()

                        # Compute checksum
                        checksum = hashlib.sha256(file_bytes).hexdigest()

                        # Determine MIME type
                        mime_type = get_mime_type(file_path.name)

                        # Compute expiry
                        expires_at = None
                        if ttl_hours > 0:
                            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

                        # Create SandboxFile record
                        file_id = uuid.uuid4()
                        sandbox_file = SandboxFile(
                            id=file_id,
                            session_id=session_id,
                            filename=file_path.name,
                            mime_type=mime_type,
                            file_size=file_size,
                            checksum=checksum,
                            content=file_bytes,
                            expires_at=expires_at,
                        )
                        db_session.add(sandbox_file)

                        # Build download URL WITH FILENAME IN PATH
                        encoded_filename = quote(file_path.name)
                        if base_url:
                            url = f"{base_url}/dl/{file_id}/{encoded_filename}"
                        else:
                            url = f"/dl/{file_id}/{encoded_filename}"

                        # Human-readable size
                        if file_size < 1024:
                            size_human = f"{file_size} B"
                        elif file_size < 1024 * 1024:
                            size_human = f"{file_size / 1024:.1f} KB"
                        else:
                            size_human = f"{file_size / (1024 * 1024):.1f} MB"

                        # Flag images for inline rendering
                        is_image = ext in IMAGE_EXTENSIONS

                        download_info.append({
                            'filename': file_path.name,
                            'url': url,
                            'size': size_human,
                            'mime_type': mime_type,
                            'file_id': str(file_id),
                            'is_image': is_image,
                            'expires_at': expires_at.isoformat() if expires_at else None,
                        })

                        logger.info(
                            f"Stored in Postgres: {file_path.name} "
                            f"({size_human}, image={is_image}) -> {url}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to store {rel_path}: {e}")
                        continue

                # Commit all files at once
                if download_info:
                    await db_session.commit()
                    logger.info(f"Committed {len(download_info)} files to Postgres")

        except Exception as e:
            logger.error(f"Failed to store files in Postgres: {e}", exc_info=True)

        return download_info

    # ================================================================
    # Main Execution
    # ================================================================

    async def execute(
        self,
        code: str,
        session_id: str = "default",
        timeout: int = None,
        context: Dict = None
    ) -> ExecutionResult:
        """Execute Python code in sandbox with PERSISTENT STATE.

        Variables, dataframes, and imports persist across calls
        within the same session_id. Like a Jupyter notebook.

        Generated files are automatically stored in Postgres and
        download URLs are included in the result.

        Charts created with matplotlib are auto-captured when plt.show()
        is called. The resulting PNG URLs are returned as inline_images
        for rendering directly in the chat.

        Args:
            code: Python code to execute
            session_id: Session ID for state + file isolation
            timeout: Max execution time in seconds
            context: Additional variables to inject

        Returns:
            ExecutionResult with stdout, stderr, result, download_urls,
            inline_images, etc.
        """
        result = ExecutionResult()
        timeout = timeout or settings.MAX_EXECUTION_TIME

        logger.info(f"Executing code: session={session_id}, timeout={timeout}s")
        logger.info(f"Code preview: {code[:200]}")

        # Create session directory
        session_dir = self.sandbox_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # PERSISTENT KERNEL: Check if kernel exists BEFORE building globals
        # =====================================================================
        try:
            sandbox_globals = kernel_manager.get_existing(session_id)
            
            if sandbox_globals is not None:
                logger.info(f"Fast path: reusing existing kernel for session={session_id}")
            else:
                logger.info(f"Slow path: building fresh globals for session={session_id}")
                fresh_globals = self._build_safe_globals(session_dir)
                sandbox_globals = kernel_manager.get_or_create(
                    session_id=session_id,
                    sandbox_globals=fresh_globals,
                    session_dir=session_dir,
                )
        except Exception as e:
            logger.error(f"Failed to get/create kernel: {e}", exc_info=True)
            result.success = False
            result.error_message = f"Kernel initialization failed: {e}"
            result.error_traceback = traceback.format_exc()
            return result

        # Reset RESULT for this execution (don't carry over from last call)
        sandbox_globals['RESULT'] = None

        # Inject context variables
        if context:
            sandbox_globals.update(context)

        # Preprocess code (handle imports, add safety)
        try:
            processed_code = self._preprocess_code(code, sandbox_globals)
        except Exception as e:
            logger.error(f"Failed to preprocess code: {e}", exc_info=True)
            result.success = False
            result.error_message = f"Code preprocessing failed: {e}"
            result.error_traceback = traceback.format_exc()
            return result

        logger.info(f"Processed code preview: {processed_code[:300]}")

        # =====================================================================
        # CHART CAPTURE: Set up matplotlib hooks BEFORE execution
        #
        # Create a fresh ChartCapture for each execution. Install hooks
        # on plt.show() so that any charts are saved as PNG and tracked.
        # This happens AFTER preprocessing (which may have loaded matplotlib).
        # =====================================================================
        chart_capture = ChartCapture(session_dir)
        if 'plt' in sandbox_globals or 'matplotlib' in sandbox_globals:
            self._install_chart_hooks(sandbox_globals, chart_capture)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Track files before execution
        files_before = set()
        if session_dir.exists():
            try:
                files_before = set(str(p) for p in session_dir.rglob('*') if p.is_file())
            except Exception:
                pass

        # Execute
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            def _execute():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    if HAS_RESOURCE:
                        try:
                            mem_bytes = settings.MAX_MEMORY_MB * 1024 * 1024
                            resource_module.setrlimit(
                                resource_module.RLIMIT_AS,
                                (mem_bytes, mem_bytes)
                            )
                        except (ValueError, OSError, resource_module.error) as e:
                            logger.debug(f"Could not set memory limit: {e}")

                    original_cwd = os.getcwd()
                    try:
                        os.chdir(session_dir)
                        logger.debug(f"Changed CWD to {session_dir}")

                        compiled = compile(processed_code, '<sandbox>', 'exec')
                        exec(compiled, sandbox_globals)
                    finally:
                        os.chdir(original_cwd)
                        logger.debug(f"Restored CWD to {original_cwd}")

            await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=timeout
            )

            result.success = True
            logger.info("Code execution succeeded")

            # Extract RESULT variable if set
            if 'RESULT' in sandbox_globals and sandbox_globals['RESULT'] is not None:
                result.result = sandbox_globals['RESULT']

            # Get variables from kernel session
            session_info = kernel_manager.get_session_info(session_id)
            if session_info:
                result.variables = session_info['variables']
                result.kernel_info = {
                    'execution_count': session_info['execution_count'],
                    'variable_count': session_info['variable_count'],
                    'session_persisted': True,
                }

        except asyncio.TimeoutError:
            result.success = False
            result.error_message = f"Execution timed out after {timeout} seconds"
            result.error_traceback = ""
            logger.warning(f"Execution timed out after {timeout}s")

        except MemoryError:
            result.success = False
            result.error_message = f"Memory limit exceeded ({settings.MAX_MEMORY_MB} MB)"
            result.error_traceback = ""
            logger.warning("Memory limit exceeded")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            logger.error(f"Execution error: {e}", exc_info=True)

        finally:
            end_time = time.time()
            result.execution_time_ms = int((end_time - start_time) * 1000)
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            logger.info(f"Execution completed: success={result.success}, "
                       f"time={result.execution_time_ms}ms, "
                       f"stdout_len={len(result.stdout)}, "
                       f"stderr_len={len(result.stderr)}, "
                       f"charts_captured={len(chart_capture.captured_charts)}")

            if result.stdout:
                logger.info(f"stdout preview: {result.stdout[:200]}")
            if result.error_message:
                logger.error(f"error: {result.error_message}")

            # Track memory usage
            if HAS_RESOURCE:
                try:
                    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
                    result.memory_used_mb = usage.ru_maxrss / 1024
                except Exception:
                    result.memory_used_mb = 0.0

            # Track new files created (includes chart PNGs from auto-capture)
            if session_dir.exists():
                try:
                    files_after = set(str(p) for p in session_dir.rglob('*') if p.is_file())
                    new_files = files_after - files_before
                    result.files_created = [
                        str(Path(f).relative_to(session_dir)) for f in new_files
                    ]
                    if result.files_created:
                        logger.info(f"New files detected: {result.files_created}")
                except Exception:
                    pass

        # =================================================================
        # AUTO FILE STORAGE + INLINE IMAGE DETECTION
        #
        # Store all new files in Postgres. For images (especially charts),
        # also populate inline_images for rendering in chat.
        # =================================================================
        if result.files_created:
            try:
                download_info = await self._store_files_in_postgres(
                    new_file_paths=result.files_created,
                    session_id=session_id,
                    session_dir=session_dir,
                )
                result.download_urls = download_info

                # Separate images for inline rendering
                for info in download_info:
                    if info.get('is_image', False):
                        result.inline_images.append({
                            'url': info['url'],
                            'filename': info['filename'],
                            'alt_text': info['filename'].replace('_', ' ').replace('.png', ''),
                        })

                # Log chart capture results
                if chart_capture.captured_charts:
                    logger.info(
                        f"Charts auto-captured: {chart_capture.captured_charts} "
                        f"-> {len(result.inline_images)} inline images"
                    )

                # Append download URLs to stdout for non-image files
                non_image_downloads = [d for d in download_info if not d.get('is_image', False)]
                if non_image_downloads:
                    url_lines = ["\n\nGenerated files ready for download:"]
                    for info in non_image_downloads:
                        url_lines.append(
                            f"\n[{info['filename']} ({info['size']}) - Click to Download]({info['url']})"
                        )
                        url_lines.append(f"Direct link: {info['url']}")
                    url_summary = '\n'.join(url_lines)
                    result.stdout = result.stdout + url_summary
                    logger.info(f"Appended {len(non_image_downloads)} download URLs to stdout")

                # For images, append markdown image syntax to stdout
                # This gives the AI agent the URLs to render inline
                if result.inline_images:
                    img_lines = ["\n\nGenerated charts:"]
                    for img in result.inline_images:
                        img_lines.append(f"\n![{img['alt_text']}]({img['url']})")
                    result.stdout = result.stdout + '\n'.join(img_lines)
                    logger.info(f"Appended {len(result.inline_images)} inline image URLs to stdout")

            except Exception as e:
                logger.error(f"File storage failed (non-fatal): {e}", exc_info=True)

        return result


# Singleton executor
executor = SandboxExecutor()
