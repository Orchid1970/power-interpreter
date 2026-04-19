"""Power Interpreter - Sandboxed Code Executor
Version: see app/version.py
"""

import asyncio
import io
import os
import sys
import time
import traceback
import tempfile
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
from contextlib import redirect_stdout, redirect_stderr
import logging
import uuid
import re as re_module
from urllib.parse import quote

from app.config import settings
from app.engine.kernel_manager import kernel_manager
from app.engine.session_store import session_store
from app.syntax_guard import check_syntax as _syntax_check
from app.context_guard import truncate_stdout as _truncate_stdout

logger = logging.getLogger(__name__)

MIN_EXECUTION_TIMEOUT = 100

# Module-level single-capture of the TRUE pandas originals. Initialised lazily
# the first time _install_pandas_path_hooks runs. Without this, every per-call
# wrap treated the already-wrapped function as the "original", producing
# N-deep recursion chains (observed up to 13 deep in production traces).
_PANDAS_ORIGINAL_READ_CSV = None
_PANDAS_ORIGINAL_READ_EXCEL = None

try:
    import resource as resource_module
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

STORABLE_EXTENSIONS = {
    '.xlsx', '.xls', '.csv', '.tsv', '.json',
    '.pdf', '.png', '.jpg', '.jpeg', '.svg',
    '.html', '.txt', '.md', '.zip', '.parquet',
    '.docx', '.doc',
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg'}

MATPLOTLIB_USE_PATTERN = re_module.compile(
    r'^\s*matplotlib\s*\.\s*use\s*\(\s*[\'\"][^\'\"]*[\'\"]\s*\)\s*$',
    re_module.MULTILINE
)

# Deprecated pandas frequency aliases (pandas 2.x)
DEPRECATED_FREQ_ALIASES = {
    'M': 'ME',
    'Y': 'YE',
    'A': 'YE',
    'Q': 'QE',
    'H': 'h',
    'T': 'min',
    'S': 's',
    'L': 'ms',
    'U': 'us',
    'N': 'ns',
    'BM': 'BME',
    'BY': 'BYE',
    'BQ': 'BQE',
    'BA': 'BYE',
}

REDIRECT_PATH_PREFIXES = ('/tmp/', '/temp/', '/var/tmp/')

ALLOWED_READ_PATHS = [
    '/home/ubuntu/uploads',
    '/home/ubuntu/uploads/tmp',
    '/home/ubuntu/uploads/tmp/permanent_files',
    '/app/uploads',
    '/uploads',
    '/app/sandbox_data',
]

LEGITIMATE_READ_PREFIXES = (
    '/home/ubuntu/uploads/',
    '/app/uploads/',
    '/uploads/',
    '/app/sandbox_data/',
)


class ExecutionResult:
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
        self.download_urls: list = []
        self.inline_images: list = []
        self.variables: Dict[str, str] = {}
        self.kernel_info: Dict[str, Any] = {}

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
        if self.result is None:
            return None
        try:
            import json
            json.dumps(self.result)
            return self.result
        except (TypeError, ValueError):
            return str(self.result)[:settings.MAX_OUTPUT_SIZE]


# =============================================================================
# CHART CAPTURE - Fixed: make_savefig_wrapper is now a proper class method
# =============================================================================

class ChartCapture:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.captured_charts: List[str] = []
        self._chart_counter = 0
        self._savefig_tracked: set = set()

    def make_show_replacement(self, plt_module):
        capture = self

        def _capturing_show(*args, **kwargs):
            try:
                fig_nums = plt_module.get_fignums()
                if not fig_nums:
                    return
                for fig_num in fig_nums:
                    fig = plt_module.figure(fig_num)
                    capture._chart_counter += 1
                    chart_name = f"chart_{capture._chart_counter:03d}.png"
                    chart_path = capture.session_dir / chart_name
                    fig.savefig(str(chart_path), format='png', dpi=150,
                                bbox_inches='tight', facecolor='white',
                                edgecolor='none', pad_inches=0.1)
                    capture.captured_charts.append(chart_name)
                plt_module.close('all')
            except Exception as e:
                logger.error(f"Chart capture via plt.show() failed: {e}")

        return _capturing_show

    def make_savefig_wrapper(self, original_savefig):
        """Wrap Figure.savefig to track files saved by user code.

        FIX: This was previously nested inside capture_unclosed_figures
        due to an indentation error, making it inaccessible as a class method.
        Now it's a proper method on ChartCapture.
        """
        capture = self

        def _tracking_savefig(self_fig, fname, *args, **kwargs):
            result = original_savefig(self_fig, fname, *args, **kwargs)
            try:
                fname_str = str(fname)
                fname_path = Path(fname_str)
                ext = fname_path.suffix.lower()

                if ext in IMAGE_EXTENSIONS or ext in STORABLE_EXTENSIONS:
                    if not fname_path.is_absolute():
                        rel_name = fname_str
                    else:
                        try:
                            rel_name = str(fname_path.relative_to(capture.session_dir))
                        except ValueError:
                            # File saved outside session dir - copy it in
                            if fname_path.exists():
                                dest = capture.session_dir / fname_path.name
                                shutil.copy2(str(fname_path), str(dest))
                                rel_name = fname_path.name
                                logger.info(f"savefig: copied {fname_path} -> {dest}")
                            else:
                                rel_name = fname_str

                    if rel_name not in capture._savefig_tracked:
                        capture._savefig_tracked.add(rel_name)
                        logger.info(f"savefig tracked: {rel_name}")
                        if ext in IMAGE_EXTENSIONS:
                            capture.captured_charts.append(rel_name)
            except Exception as e:
                logger.warning(f"savefig tracking failed: {e}")
            return result

        return _tracking_savefig

    def capture_unclosed_figures(self, plt_module):
        """Auto-capture any matplotlib figures that weren't explicitly saved or shown."""
        try:
            fig_nums = plt_module.get_fignums()
            if not fig_nums:
                return
            for fig_num in fig_nums:
                fig = plt_module.figure(fig_num)
                self._chart_counter += 1
                chart_name = f"chart_{self._chart_counter:03d}.png"
                chart_path = self.session_dir / chart_name
                if chart_name in self.captured_charts:
                    continue
                fig.savefig(str(chart_path), format='png', dpi=150,
                            bbox_inches='tight', facecolor='white',
                            edgecolor='none', pad_inches=0.1)
                self.captured_charts.append(chart_name)
            plt_module.close('all')
        except Exception as e:
            logger.error(f"Auto-capture of unclosed figures failed: {e}")


def _is_allowed_read_path(filepath: str) -> bool:
    try:
        resolved = os.path.realpath(filepath)
        return any(resolved.startswith(prefix) for prefix in LEGITIMATE_READ_PREFIXES)
    except Exception:
        return False


def _redirect_path(filepath: str, session_dir: Path) -> str:
    if any(filepath.startswith(prefix) for prefix in REDIRECT_PATH_PREFIXES):
        filename = os.path.basename(filepath)
        return str(session_dir / filename)
    return filepath


# =============================================================================
# SESSION SEQUENCER - Ensures ordered execution of numbered steps
# =============================================================================

class SessionSequencer:
    """Manages ordered execution of sequenced calls within a session."""

    def __init__(self):
        self._condition: Optional[asyncio.Condition] = None
        self._completed: set = set()
        self._expected_max: int = 0
        self._active_batch: bool = False

    @property
    def condition(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    async def wait_for_turn(self, sequence: int, timeout: float = 120.0):
        if sequence == 1:
            self._completed.clear()
            self._expected_max = 0

        self._active_batch = True
        self._expected_max = max(self._expected_max, sequence)

        if sequence == 1:
            return

        async with self.condition:
            try:
                await asyncio.wait_for(
                    self.condition.wait_for(
                        lambda: all(i in self._completed for i in range(1, sequence))
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Sequence {sequence} timed out waiting for predecessors. "
                    f"Completed: {sorted(self._completed)}. Proceeding anyway."
                )

    async def mark_completed(self, sequence: int):
        self._completed.add(sequence)
        async with self.condition:
            self.condition.notify_all()

        if self._expected_max > 0 and len(self._completed) >= self._expected_max:
            self._active_batch = False

    async def wait_for_batch_clear(self, timeout: float = 120.0):
        if not self._active_batch:
            return
        async with self.condition:
            try:
                await asyncio.wait_for(
                    self.condition.wait_for(lambda: not self._active_batch),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Non-sequenced call timed out waiting for batch. Proceeding.")


class SandboxExecutor:
    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or settings.SANDBOX_DIR
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._async_locks: Dict[str, asyncio.Lock] = {}
        self._sequencers: Dict[str, SessionSequencer] = {}

    def _get_async_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._async_locks:
            self._async_locks[session_id] = asyncio.Lock()
        return self._async_locks[session_id]

    def _get_sequencer(self, session_id: str) -> SessionSequencer:
        if session_id not in self._sequencers:
            self._sequencers[session_id] = SessionSequencer()
        return self._sequencers[session_id]

    def _get_safe_builtins(self) -> Dict:
        safe = {}
        import builtins as builtins_module

        blocked = set(settings.BLOCKED_BUILTINS) if hasattr(settings, 'BLOCKED_BUILTINS') else {
            'eval', 'exec', 'compile', 'globals', 'locals',
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

        safe['__import__'] = __import__
        safe['__build_class__'] = getattr(builtins_module, '__build_class__', None)

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

        # Explicitly whitelist open() - some minimal container builtins dicts
        # strip it, causing NameError on any file write from sandboxed code.
        # Forcing the reference here is a no-op where it was already present.
        safe['open'] = builtins_module.open

        return safe

    def _lazy_import(self, module_name: str, alias: str = None):
        actual_alias = alias or module_name
        try:
            if module_name == 'pandas':
                import pandas as pd
                return pd
            elif module_name == 'numpy':
                import numpy as np
                return np
            elif module_name == 'matplotlib.pyplot':
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                return plt
            elif module_name == 'matplotlib':
                import matplotlib
                matplotlib.use('Agg')
                return matplotlib
            elif module_name == 'seaborn':
                import seaborn as sns
                return sns
            elif module_name == 'plotly.express':
                import plotly.express as px
                return px
            elif module_name == 'plotly.graph_objects':
                import plotly.graph_objects as go
                return go
            elif module_name == 'scipy':
                import scipy
                return scipy
            elif module_name == 'sklearn':
                import sklearn
                return sklearn
            elif module_name == 'statsmodels':
                import statsmodels
                return statsmodels
            elif module_name == 'statsmodels.api':
                import statsmodels.api as sm
                return sm
            elif module_name == 'openpyxl':
                import openpyxl
                return openpyxl
            elif module_name == 'pdfplumber':
                import pdfplumber
                return pdfplumber
            elif module_name == 'tabulate':
                import tabulate
                return tabulate
            elif module_name == 'xlsxwriter':
                import xlsxwriter
                return xlsxwriter
            elif module_name == 'json':
                import json
                return json
            elif module_name == 'csv':
                import csv
                return csv
            elif module_name == 'math':
                import math
                return math
            elif module_name == 'statistics':
                import statistics
                return statistics
            elif module_name == 'datetime':
                import datetime
                return datetime
            elif module_name == 'collections':
                import collections
                return collections
            elif module_name == 'itertools':
                import itertools
                return itertools
            elif module_name == 'functools':
                import functools
                return functools
            elif module_name == 're':
                import re
                return re
            elif module_name == 'io':
                import io
                return io
            elif module_name == 'copy':
                import copy
                return copy
            elif module_name == 'hashlib':
                import hashlib
                return hashlib
            elif module_name == 'base64':
                import base64
                return base64
            elif module_name == 'textwrap':
                import textwrap
                return textwrap
            elif module_name == 'string':
                import string
                return string
            elif module_name == 'struct':
                import struct
                return struct
            elif module_name == 'decimal':
                import decimal
                return decimal
            elif module_name == 'fractions':
                import fractions
                return fractions
            elif module_name == 'random':
                import random
                return random
            elif module_name == 'time':
                import time
                return time
            elif module_name == 'calendar':
                import calendar
                return calendar
            elif module_name == 'pprint':
                import pprint
                return pprint
            elif module_name == 'dataclasses':
                import dataclasses
                return dataclasses
            elif module_name == 'typing':
                import typing
                return typing
            elif module_name == 'pathlib':
                import pathlib
                return pathlib
            elif module_name == 'os':
                import os as os_module
                return os_module
            elif module_name == 'sys':
                import sys as sys_module
                return sys_module
            elif module_name == 'urllib':
                import urllib
                return urllib
            elif module_name == 'requests':
                import requests
                return requests
            else:
                import importlib
                return importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Module {module_name} not available: {e}")
            return None

    def _build_safe_globals(self, session_dir: Path) -> Dict[str, Any]:
        sandbox_globals = {
            '__builtins__': self._get_safe_builtins(),
            '__name__': '__main__',
            'SANDBOX_DIR': str(session_dir),
            'RESULT': None,
        }

        import_map = {
            'pd': 'pandas', 'pandas': 'pandas',
            'np': 'numpy', 'numpy': 'numpy',
            'json': 'json', 'csv': 'csv',
            'math': 'math', 'statistics': 'statistics',
            'datetime': 'datetime', 'collections': 'collections',
            'itertools': 'itertools', 'functools': 'functools',
            're': 're', 'io': 'io', 'copy': 'copy',
            'hashlib': 'hashlib', 'base64': 'base64',
            'textwrap': 'textwrap', 'string': 'string',
            'struct': 'struct', 'decimal': 'decimal',
            'fractions': 'fractions', 'random': 'random',
            'time': 'time', 'calendar': 'calendar',
            'pprint': 'pprint', 'dataclasses': 'dataclasses',
            'typing': 'typing', 'pathlib': 'pathlib',
            'os': 'os', 'sys': 'sys',
        }

        for alias, module_name in import_map.items():
            mod = self._lazy_import(module_name, alias)
            if mod is not None:
                sandbox_globals[alias] = mod

        from pathlib import Path as PathClass
        from dataclasses import dataclass, field, asdict
        from decimal import Decimal
        from fractions import Fraction
        from typing import Dict, List, Optional, Tuple, Set, Any

        sandbox_globals.update({
            'Path': PathClass,
            'dataclass': dataclass,
            'field': field,
            'asdict': asdict,
            'Decimal': Decimal,
            'Fraction': Fraction,
            'Dict': Dict,
            'List': List,
            'Optional': Optional,
            'Tuple': Tuple,
            'Set': Set,
            'Any': Any,
        })

        return sandbox_globals

    def _patch_common_llm_mistakes(self, code: str) -> str:
        patched = code

        def _fix_freq_arg(m):
            prefix = m.group(1)
            quote_char = m.group(2)
            alias = m.group(3)
            if alias in DEPRECATED_FREQ_ALIASES:
                replacement = DEPRECATED_FREQ_ALIASES[alias]
                logger.debug(f"LLM patch: freq='{alias}' -> freq='{replacement}'")
                return f"{prefix}{quote_char}{replacement}{quote_char}"
            return m.group(0)

        patched = re_module.sub(
            r"""(freq\s*=\s*)(['\"])([A-Z]{1,2})\2""",
            _fix_freq_arg,
            patched
        )

        def _fix_resample_arg(m):
            prefix = m.group(1)
            quote_char = m.group(2)
            alias = m.group(3)
            suffix = m.group(4)
            if alias in DEPRECATED_FREQ_ALIASES:
                replacement = DEPRECATED_FREQ_ALIASES[alias]
                logger.debug(f"LLM patch: resample('{alias}') -> resample('{replacement}')")
                return f"{prefix}{quote_char}{replacement}{quote_char}{suffix}"
            return m.group(0)

        patched = re_module.sub(
            r"""(\.resample\(\s*)(['\"])([A-Z]{1,2})\2(\s*\))""",
            _fix_resample_arg,
            patched
        )

        def _fix_asfreq_arg(m):
            prefix = m.group(1)
            quote_char = m.group(2)
            alias = m.group(3)
            suffix = m.group(4)
            if alias in DEPRECATED_FREQ_ALIASES:
                replacement = DEPRECATED_FREQ_ALIASES[alias]
                logger.debug(f"LLM patch: asfreq('{alias}') -> asfreq('{replacement}')")
                return f"{prefix}{quote_char}{replacement}{quote_char}{suffix}"
            return m.group(0)

        patched = re_module.sub(
            r"""(\.asfreq\(\s*)(['\"])([A-Z]{1,2})\2(\s*\))""",
            _fix_asfreq_arg,
            patched
        )

        if 'urllib.request.request(' in patched:
            patched = patched.replace('urllib.request.request(', 'urllib.request.urlopen(')
            logger.debug("LLM patch: urllib.request.request() -> urllib.request.urlopen()")

        return patched

    def _preprocess_code(self, code: str, sandbox_globals: Dict) -> str:
        code = MATPLOTLIB_USE_PATTERN.sub('# matplotlib.use() handled by sandbox', code)

        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('#') or not stripped:
                processed_lines.append(line)
                continue

            if stripped.startswith('import ') or stripped.startswith('from '):
                handled = self._handle_import_line(stripped, sandbox_globals)
                if handled:
                    indent = line[:len(line) - len(line.lstrip())]
                    processed_lines.append(f"{indent}{handled}")
                    continue

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    def _handle_import_line(self, line: str, sandbox_globals: Dict) -> Optional[str]:
        if line.startswith('import '):
            parts = line[7:].split(',')
            assignments = []
            for part in parts:
                part = part.strip()
                if ' as ' in part:
                    module_name, alias = part.split(' as ', 1)
                    module_name = module_name.strip()
                    alias = alias.strip()
                else:
                    module_name = part
                    alias = part.split('.')[0]

                if alias not in sandbox_globals:
                    mod = self._lazy_import(module_name, alias)
                    if mod is not None:
                        sandbox_globals[alias] = mod
                assignments.append(f"{alias} = {alias}")
            return '; '.join(assignments) if assignments else '# import handled'

        elif line.startswith('from '):
            match = re_module.match(r'from\s+([\w.]+)\s+import\s+(.+)', line)
            if not match:
                return None

            module_name = match.group(1)
            imports_str = match.group(2).strip()

            if module_name not in sandbox_globals:
                mod = self._lazy_import(module_name)
                if mod is not None:
                    sandbox_globals[module_name] = mod

            mod = sandbox_globals.get(module_name) or self._lazy_import(module_name)
            if mod is None:
                return f"# Module {module_name} not available"

            if imports_str == '*':
                if hasattr(mod, '__all__'):
                    for name in mod.__all__:
                        if hasattr(mod, name):
                            sandbox_globals[name] = getattr(mod, name)
                return f"# from {module_name} import * handled"

            assignments = []
            for item in imports_str.split(','):
                item = item.strip()
                if ' as ' in item:
                    name, alias = item.split(' as ', 1)
                    name = name.strip()
                    alias = alias.strip()
                else:
                    name = item
                    alias = item

                if hasattr(mod, name):
                    sandbox_globals[alias] = getattr(mod, name)
                    assignments.append(f"{alias} = {alias}")
                else:
                    resolved = False
                    try:
                        submod = self._lazy_import(f"{module_name}.{name}")
                        if submod is not None:
                            sandbox_globals[alias] = submod
                            assignments.append(f"{alias} = {alias}")
                            resolved = True
                    except Exception:
                        pass

                    if not resolved:
                        try:
                            import importlib
                            parent = importlib.import_module(module_name)
                            if hasattr(parent, name):
                                sandbox_globals[alias] = getattr(parent, name)
                                assignments.append(f"{alias} = {alias}")
                            else:
                                assignments.append(f"# {name} not found in {module_name}")
                        except Exception:
                            assignments.append(f"# {name} not found in {module_name}")
            return '; '.join(assignments) if assignments else f"# from {module_name} import handled"

        return None

    def _install_chart_hooks(self, sandbox_globals: Dict, chart_capture: ChartCapture):
        plt_module = sandbox_globals.get('plt')
        if plt_module:
            sandbox_globals['plt'].show = chart_capture.make_show_replacement(plt_module)

        matplotlib_module = sandbox_globals.get('matplotlib')
        if matplotlib_module and hasattr(matplotlib_module, 'pyplot'):
            matplotlib_module.pyplot.show = chart_capture.make_show_replacement(matplotlib_module.pyplot)

        try:
            import matplotlib.figure
            original_savefig = matplotlib.figure.Figure.savefig
            matplotlib.figure.Figure.savefig = chart_capture.make_savefig_wrapper(original_savefig)
        except Exception as e:
            logger.warning(f"Failed to install savefig hook: {e}")

    def _make_path_normalizer(self, session_dir: Path):
        def _normalize_path(filepath: str) -> str:
            return _redirect_path(filepath, session_dir)
        return _normalize_path

    def _install_pandas_path_hooks(self, sandbox_globals: Dict, session_dir: Path):
        global _PANDAS_ORIGINAL_READ_CSV, _PANDAS_ORIGINAL_READ_EXCEL

        pd_module = sandbox_globals.get('pd')
        if not pd_module:
            return

        # Capture the TRUE originals exactly once (module-level). Previously each
        # call treated the already-wrapped function as the "original", producing
        # a recursion chain that grew one level per execute() call - observed up
        # to 13 levels in production traces before this fix.
        if _PANDAS_ORIGINAL_READ_CSV is None:
            _PANDAS_ORIGINAL_READ_CSV = pd_module.read_csv
        if _PANDAS_ORIGINAL_READ_EXCEL is None:
            _PANDAS_ORIGINAL_READ_EXCEL = pd_module.read_excel

        original_read_csv = _PANDAS_ORIGINAL_READ_CSV
        original_read_excel = _PANDAS_ORIGINAL_READ_EXCEL

        def _resolve_path(filepath_or_buffer):
            """Path resolution for pandas reads with a three-step fallback:
              1. Redirect /tmp/* (and similar) to session_dir - matches the
                 write path so round-trips stay symmetric.
              2. If relative, try session_dir / path.
              3. If still not found, look for basename in session_dir. Handles
                 the write/read asymmetry where code writes to '/tmp/foo.csv'
                 (redirected to session_dir) then reads back via the /tmp path.
            Non-string inputs (buffers, URLs handled by pandas) pass through.
            """
            if not isinstance(filepath_or_buffer, str):
                return filepath_or_buffer

            redirected = _redirect_path(filepath_or_buffer, session_dir)

            if not os.path.isabs(redirected):
                candidate = session_dir / redirected
                if candidate.exists():
                    return str(candidate)

            if not os.path.exists(redirected):
                fallback = session_dir / os.path.basename(redirected)
                if fallback.exists():
                    return str(fallback)

            return redirected

        def _patched_read_csv(filepath_or_buffer, *args, **kwargs):
            return original_read_csv(_resolve_path(filepath_or_buffer), *args, **kwargs)

        def _patched_read_excel(filepath_or_buffer, *args, **kwargs):
            return original_read_excel(_resolve_path(filepath_or_buffer), *args, **kwargs)

        sandbox_globals['pd'].read_csv = _patched_read_csv
        sandbox_globals['pd'].read_excel = _patched_read_excel

    async def _store_files_in_postgres(self, new_file_paths: list, session_id: str,
                                        session_dir: Path) -> list:
        download_info = []
        try:
            from app.database import get_session_factory
            from app.models import SandboxFile

            factory = get_session_factory()
            async with factory() as db_session:
                for rel_path in new_file_paths:
                    full_path = session_dir / rel_path
                    if not full_path.exists():
                        continue

                    ext = full_path.suffix.lower()
                    if ext not in STORABLE_EXTENSIONS:
                        continue

                    file_data = full_path.read_bytes()
                    file_size = len(file_data)
                    file_id = uuid.uuid4()

                    mime_type_map = {
                        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                        '.svg': 'image/svg+xml', '.pdf': 'application/pdf',
                        '.csv': 'text/csv', '.json': 'application/json',
                        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        '.xls': 'application/vnd.ms-excel',
                        '.html': 'text/html', '.txt': 'text/plain', '.md': 'text/markdown',
                        '.zip': 'application/zip', '.parquet': 'application/octet-stream',
                        '.tsv': 'text/tab-separated-values',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.doc': 'application/msword',
                    }
                    mime_type = mime_type_map.get(ext, 'application/octet-stream')

                    now_utc = datetime.utcnow()

                    sandbox_file = SandboxFile(
                        id=file_id,
                        session_id=session_id,
                        filename=full_path.name,
                        mime_type=mime_type,
                        file_size=file_size,
                        content=file_data,
                        checksum=hashlib.sha256(file_data).hexdigest(),
                        created_at=now_utc,
                        expires_at=now_utc + timedelta(hours=24),
                    )
                    db_session.add(sandbox_file)

                    from app.config import settings as app_settings
                    base_url = app_settings.public_base_url
                    filename_encoded = quote(full_path.name)
                    download_url = f"{base_url}/dl/{file_id}/{filename_encoded}"

                    is_image = ext in IMAGE_EXTENSIONS

                    if file_size >= 1024 * 1024:
                        size_human = f"{file_size / (1024 * 1024):.1f} MB"
                    elif file_size >= 1024:
                        size_human = f"{file_size / 1024:.1f} KB"
                    else:
                        size_human = f"{file_size} B"

                    download_info.append({
                        'filename': full_path.name,
                        'url': download_url,
                        'size': size_human,
                        'file_id': file_id,
                        'is_image': is_image,
                    })

                await db_session.commit()
                logger.info(f"Stored {len(download_info)} files in Postgres for session={session_id}")

        except Exception as e:
            logger.error(f"Postgres file storage failed: {e}", exc_info=True)
            download_info = []

        return download_info

    async def execute(self, code: str, session_id: str = "default",
                      timeout: int = None, context: Dict = None,
                      sequence: Optional[int] = None) -> ExecutionResult:
        result = ExecutionResult()

        raw_timeout = timeout
        timeout = max(timeout or settings.MAX_EXECUTION_TIME, MIN_EXECUTION_TIMEOUT)
        if raw_timeout and raw_timeout < MIN_EXECUTION_TIMEOUT:
            logger.debug(f"Timeout floor: {raw_timeout}s -> {MIN_EXECUTION_TIMEOUT}s")

        session_dir = self.sandbox_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Register / refresh session in the async-native SessionStore BEFORE
        # we acquire the per-session exec lock, so a long-running prior
        # request still keeps the session alive against the TTL sweeper.
        try:
            await session_store.touch(session_id)
        except Exception as e:
            logger.warning(f"session_store.touch failed (non-fatal): {e}")

        effective_sequence = sequence if (sequence and sequence > 0) else None

        sequencer = self._get_sequencer(session_id)

        if effective_sequence is not None:
            logger.info(f"execute: session={session_id}, sequence={effective_sequence}, waiting for turn")
            await sequencer.wait_for_turn(effective_sequence, timeout=float(timeout))
        else:
            await sequencer.wait_for_batch_clear(timeout=30.0)

        async_lock = self._get_async_lock(session_id)
        async with async_lock:
            logger.info(f"execute: session={session_id}, "
                        f"sequence={effective_sequence or 'none'}, timeout={timeout}s")
            try:
                return await self._execute_locked(
                    code, session_id, session_dir, timeout, result, context
                )
            finally:
                if effective_sequence is not None:
                    await sequencer.mark_completed(effective_sequence)

    async def _execute_locked(self, code: str, session_id: str, session_dir: Path,
                               timeout: int, result: ExecutionResult,
                               context: Dict = None) -> ExecutionResult:

        if kernel_manager.has_session(session_id):
            sandbox_globals = kernel_manager.get_or_create(
                session_id, {}, session_dir
            )
            logger.debug(f"KERNEL_DIAG: REUSED session={session_id}, "
                         f"user_vars={list(kernel_manager.get_session_info(session_id)['variables'].keys())}, "
                         f"total_keys={len(sandbox_globals)}")
        else:
            sandbox_globals = self._build_safe_globals(session_dir)
            sandbox_globals = kernel_manager.get_or_create(
                session_id, sandbox_globals, session_dir
            )
            logger.debug(f"KERNEL_DIAG: CREATED session={session_id}, "
                         f"total_keys={len(sandbox_globals)}")

        sandbox_globals['__normalize_path'] = self._make_path_normalizer(session_dir)
        self._install_pandas_path_hooks(sandbox_globals, session_dir)

        # -- Pre-execution syntax guard --
        syntax_error = _syntax_check(code)
        if syntax_error:
            result.success = False
            result.error_message = syntax_error
            logger.info(f"Syntax guard rejected code: {syntax_error[:120]}")
            return result

        try:
            code = self._patch_common_llm_mistakes(code)
            processed_code = self._preprocess_code(code, sandbox_globals)
        except Exception as e:
            result.success = False
            result.error_message = f"Code preprocessing failed: {e}"
            result.error_traceback = traceback.format_exc()
            return result

        chart_capture = ChartCapture(session_dir)
        if 'plt' in sandbox_globals or 'matplotlib' in sandbox_globals:
            self._install_chart_hooks(sandbox_globals, chart_capture)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # -- Snapshot files BEFORE execution --
        files_before = set()
        tmp_files_before = set()
        if session_dir.exists():
            try:
                files_before = set(str(p) for p in session_dir.rglob('*') if p.is_file())
            except Exception:
                pass
        # Also snapshot /tmp so we can detect files saved there
        try:
            tmp_files_before = set(
                str(p) for p in Path('/tmp').iterdir()
                if p.is_file() and p.suffix.lower() in STORABLE_EXTENSIONS
            )
        except Exception:
            pass

        # Snapshot the process CWD separately so the CWD rescue diff later only
        # picks up files genuinely created during this execution. Previously the
        # rescue compared against files_before (which only covers session_dir),
        # so every storable-extension file in CWD (README.md, requirements.txt,
        # etc.) was being scooped into the session on each run.
        cwd_files_before = set()
        try:
            cwd_path_pre = Path(os.getcwd())
            if cwd_path_pre.resolve() != session_dir.resolve():
                cwd_files_before = set(
                    str(p) for p in cwd_path_pre.iterdir()
                    if p.is_file() and p.suffix.lower() in STORABLE_EXTENSIONS
                )
        except Exception:
            pass

        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            def _execute():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    if HAS_RESOURCE:
                        try:
                            mem_bytes = settings.MAX_MEMORY_MB * 1024 * 1024
                            resource_module.setrlimit(resource_module.RLIMIT_AS, (mem_bytes, mem_bytes))
                        except (ValueError, OSError, resource_module.error):
                            pass

                    original_cwd = os.getcwd()
                    try:
                        os.chdir(session_dir)
                        compiled = compile(processed_code, '<sandbox>', 'exec')
                        exec(compiled, sandbox_globals)
                    finally:
                        os.chdir(original_cwd)

            await asyncio.wait_for(loop.run_in_executor(None, _execute), timeout=timeout)

            result.success = True

            if 'plt' in sandbox_globals:
                try:
                    chart_capture.capture_unclosed_figures(sandbox_globals['plt'])
                except Exception:
                    pass

            if 'RESULT' in sandbox_globals and sandbox_globals['RESULT'] is not None:
                result.result = sandbox_globals['RESULT']

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
            result.error_message = f"Execution timed out after {timeout}s"

        except MemoryError:
            result.success = False
            result.error_message = f"Memory limit exceeded ({settings.MAX_MEMORY_MB} MB)"

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()

        finally:
            end_time = time.time()
            result.execution_time_ms = int((end_time - start_time) * 1000)
            result.stdout = _truncate_stdout(stdout_capture.getvalue())
            result.stderr = stderr_capture.getvalue()

            if result.error_message:
                logger.error(f"error: {result.error_message}")

            if HAS_RESOURCE:
                try:
                    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
                    result.memory_used_mb = usage.ru_maxrss / 1024
                except Exception:
                    result.memory_used_mb = 0.0

            # -- File detection: compare before/after in session_dir --
            new_files_from_diff = set()
            if session_dir.exists():
                try:
                    files_after = set(str(p) for p in session_dir.rglob('*') if p.is_file())
                    new_files_from_diff = files_after - files_before
                except Exception:
                    pass

            # -- Savefig-tracked files: add any that the diff missed --
            savefig_files = set()
            for tracked in chart_capture._savefig_tracked:
                tracked_path = Path(tracked)
                if not tracked_path.is_absolute():
                    tracked_path = session_dir / tracked_path
                if tracked_path.exists():
                    savefig_files.add(str(tracked_path))

            # -- RESCUE: Check /tmp for new storable files --
            tmp_rescued = set()
            try:
                tmp_files_after = set(
                    str(p) for p in Path('/tmp').iterdir()
                    if p.is_file() and p.suffix.lower() in STORABLE_EXTENSIONS
                )
                new_tmp_files = tmp_files_after - tmp_files_before
                for tmp_file_str in new_tmp_files:
                    tmp_path = Path(tmp_file_str)
                    if tmp_path.exists() and tmp_path.stat().st_size > 0:
                        dest = session_dir / tmp_path.name
                        # Avoid overwriting if name collision
                        if dest.exists():
                            stem = dest.stem
                            ext = dest.suffix
                            dest = session_dir / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
                        shutil.copy2(str(tmp_path), str(dest))
                        tmp_rescued.add(str(dest))
                        logger.info(f"Rescued file from /tmp: {tmp_path.name} -> {dest.name}")
            except Exception as e:
                logger.warning(f"/tmp rescue scan failed: {e}")

            # -- RESCUE: Check CWD if it differs from session_dir --
            # Use the pre-execution cwd_files_before snapshot captured above so
            # we only pick up files genuinely created during this execution,
            # not pre-existing project files in CWD.
            cwd_rescued = set()
            try:
                current_cwd = Path(os.getcwd())
                if current_cwd.resolve() != session_dir.resolve():
                    cwd_files_after = set(
                        str(p) for p in current_cwd.iterdir()
                        if p.is_file() and p.suffix.lower() in STORABLE_EXTENSIONS
                    )
                    new_cwd_files = cwd_files_after - cwd_files_before
                    for f_str in new_cwd_files:
                        f = Path(f_str)
                        if f.exists() and f.stat().st_size > 0:
                            dest = session_dir / f.name
                            if not dest.exists():
                                shutil.copy2(str(f), str(dest))
                                cwd_rescued.add(str(dest))
                                logger.info(f"Rescued file from CWD: {f.name} -> {dest.name}")
            except Exception as e:
                logger.warning(f"CWD rescue scan failed: {e}")

            # -- Merge all detected files --
            all_new_files = new_files_from_diff | savefig_files | tmp_rescued | cwd_rescued
            result.files_created = [
                str(Path(f).relative_to(session_dir))
                for f in all_new_files
                if Path(f).suffix.lower() in STORABLE_EXTENSIONS
            ]

            logger.info(
                f"done: success={result.success}, {result.execution_time_ms}ms, "
                f"charts={len(chart_capture.captured_charts)}, "
                f"savefig_tracked={len(chart_capture._savefig_tracked)}, "
                f"diff_new={len(new_files_from_diff)}, "
                f"tmp_rescued={len(tmp_rescued)}, "
                f"cwd_rescued={len(cwd_rescued)}, "
                f"files_created={len(result.files_created)}: "
                f"{result.files_created[:5]}"
            )

        # -- Store files in Postgres and build download URLs --
        if result.files_created:
            try:
                download_info = await self._store_files_in_postgres(
                    new_file_paths=result.files_created,
                    session_id=session_id,
                    session_dir=session_dir,
                )
                result.download_urls = download_info

                for info in download_info:
                    if info.get('is_image', False):
                        result.inline_images.append({
                            'url': info['url'],
                            'filename': info['filename'],
                            'alt_text': info['filename'].replace('_', ' ').replace('.png', ''),
                        })

                non_image_downloads = [d for d in download_info if not d.get('is_image', False)]

                link_sections = []

                if non_image_downloads:
                    link_sections.append(
                        "\n\U0001F4E5 **Generated files ready for download "
                        "(present these as clickable markdown links to the user):**"
                    )
                    for info in non_image_downloads:
                        link_sections.append(
                            f"\n- [{info['filename']} ({info['size']})]({info['url']})"
                        )

                if result.inline_images:
                    link_sections.append(
                        "\n\U0001F4CA **Generated charts (embed these inline for the user):**"
                    )
                    for img in result.inline_images:
                        link_sections.append(
                            f"\n![{img['alt_text']}]({img['url']})"
                        )

                if link_sections:
                    code_output = result.stdout[:50000]
                    result.stdout = (
                        '\n'.join(link_sections) + '\n\n---\n\n' + code_output
                    )

                logger.info(
                    f"File URLs prepended: "
                    f"downloads={len(non_image_downloads)}, "
                    f"images={len(result.inline_images)}, "
                    f"stdout_len={len(result.stdout)}"
                )

            except Exception as e:
                logger.error(f"File storage failed (non-fatal): {e}", exc_info=True)

        return result


executor = SandboxExecutor()
