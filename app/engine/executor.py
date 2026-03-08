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

logger = logging.getLogger(__name__)

MIN_EXECUTION_TIMEOUT = 100

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

    def capture_unclosed_figures(self, plt_module):
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

    def make_savefig_wrapper(self, original_savefig):
        capture = self

        def _tracking_savefig(self_fig, fname, *args, **kwargs):
            result = original_savefig(self_fig, fname, *args, **kwargs)
            try:
                fname_path = Path(str(fname))
                if fname_path.suffix.lower() in IMAGE_EXTENSIONS:
                    if not fname_path.is_absolute():
                        rel_name = str(fname_path)
                        if rel_name not in capture._savefig_tracked:
                            capture._savefig_tracked.add(rel_name)
                            capture.captured_charts.append(rel_name)
            except Exception:
                pass
            return result

        return _tracking_savefig


def _is_allowed_read_path(filepath: str) -> bool:
    try:
        resolved = str(Path(filepath).resolve())
        for allowed_dir in ALLOWED_READ_PATHS:
            try:
                allowed_resolved = str(Path(allowed_dir).resolve())
                if resolved.startswith(allowed_resolved + '/') or resolved == allowed_resolved:
                    return True
            except Exception:
                if resolved.startswith(allowed_dir + '/') or resolved.startswith(allowed_dir):
                    return True
        return False
    except Exception:
        return False


def _is_legitimate_read_path(filepath: str) -> bool:
    for prefix in LEGITIMATE_READ_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


class SandboxExecutor:
    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or settings.SANDBOX_DIR
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _get_safe_builtins(self) -> Dict:
        safe = {}
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
        try:
            import pandas as pd
        except ImportError:
            pd = None

        try:
            import numpy as np
        except ImportError:
            np = None

        import json, csv, math, statistics, collections, itertools, functools, re
        import datetime as dt_module
        import io as io_module
        import copy
        import hashlib as hashlib_module
        import base64
        from decimal import Decimal
        from fractions import Fraction
        from pathlib import Path as PathLib

        try:
            from dataclasses import dataclass, field, asdict
        except ImportError:
            dataclass = field = asdict = None

        from typing import Dict, List, Optional, Tuple, Set, Any

        safe_builtins = self._get_safe_builtins()
        safe_builtins['open'] = self._make_safe_open(session_dir)

        sandbox_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            'json': json, 'csv': csv, 'math': math, 'statistics': statistics,
            'Decimal': Decimal, 'Fraction': Fraction,
            'datetime': dt_module, 'timedelta': dt_module.timedelta,
            'timezone': dt_module.timezone, 'date': dt_module.date,
            'collections': collections, 'itertools': itertools, 'functools': functools,
            're': re, 'io': io_module, 'copy': copy,
            'hashlib': hashlib_module, 'base64': base64,
            'Path': PathLib,
            'Dict': Dict, 'List': List, 'Optional': Optional,
            'Tuple': Tuple, 'Set': Set, 'Any': Any,
            'SANDBOX_DIR': str(session_dir),
            'RESULT': None,
        }

        if pd is not None:
            sandbox_globals['pd'] = pd
            sandbox_globals['pandas'] = pd
        if np is not None:
            sandbox_globals['np'] = np
            sandbox_globals['numpy'] = np
        if dataclass is not None:
            sandbox_globals['dataclass'] = dataclass
            sandbox_globals['field'] = field
            sandbox_globals['asdict'] = asdict

        return sandbox_globals

    def _normalize_path_for_sandbox(self, filepath: str, session_dir: Path) -> str:
        if not filepath:
            return filepath

        path_str = str(filepath)

        if _is_legitimate_read_path(path_str):
            return path_str

        if os.path.isabs(path_str):
            for prefix in REDIRECT_PATH_PREFIXES:
                if path_str.startswith(prefix):
                    basename = os.path.basename(path_str)
                    if basename:
                        return basename
            if len(path_str) >= 3 and path_str[1] == ':' and path_str[2] in ('\\', '/'):
                basename = os.path.basename(path_str.replace('\\', '/'))
                if basename:
                    return basename
            return path_str

        session_name = session_dir.name
        prefix_slash = session_name + '/'
        prefix_sep = session_name + os.sep

        if path_str.startswith(prefix_slash) or path_str.startswith(prefix_sep):
            stripped = path_str[len(session_name) + 1:]
            candidate = session_dir / stripped
            original = session_dir / path_str
            if candidate.exists() and not original.exists():
                return stripped
            elif not original.exists():
                return stripped

        return path_str

    def _make_safe_open(self, session_dir: Path):
        normalize = self._normalize_path_for_sandbox
        _session_dir = session_dir

        def safe_open(filepath, mode='r', *args, **kwargs):
            path_str = str(filepath)
            normalized = normalize(path_str, _session_dir)
            path = Path(normalized)
            if not path.is_absolute():
                path = _session_dir / path

            try:
                resolved = path.resolve()
                session_resolved = _session_dir.resolve()
                is_in_session = str(resolved).startswith(str(session_resolved))
                is_read_only = not any(c in mode for c in ('w', 'a', 'x', '+'))
                is_allowed_read = is_read_only and _is_allowed_read_path(str(resolved))

                if not is_in_session and not is_allowed_read:
                    if not is_read_only:
                        raise PermissionError(f"Access denied: Cannot WRITE outside sandbox. Path: {path_str}")
                    else:
                        raise PermissionError(f"Access denied: Cannot access outside sandbox. Path: {path_str}")
            except PermissionError:
                raise
            except Exception as e:
                raise PermissionError(f"Invalid file path: {e}")

            if any(c in mode for c in ('w', 'a', 'x', '+')):
                resolved.parent.mkdir(parents=True, exist_ok=True)

            return open(resolved, mode, *args, **kwargs)

        return safe_open

    def _make_path_normalizer(self, session_dir: Path):
        executor_self = self
        _session_dir = session_dir

        def _normalize_path(filepath):
            return executor_self._normalize_path_for_sandbox(str(filepath), _session_dir)

        return _normalize_path

    def _install_pandas_path_hooks(self, sandbox_globals: Dict, session_dir: Path):
        pd = sandbox_globals.get('pd')
        if pd is None:
            return

        normalizer = self._make_path_normalizer(session_dir)

        if not hasattr(pd, '_original_read_csv'):
            pd._original_read_csv = pd.read_csv

        def _patched_read_csv(filepath_or_buffer, *args, **kwargs):
            if isinstance(filepath_or_buffer, str):
                filepath_or_buffer = normalizer(filepath_or_buffer)
            return pd._original_read_csv(filepath_or_buffer, *args, **kwargs)
        sandbox_globals['pd'].read_csv = _patched_read_csv

        if not hasattr(pd, '_original_read_excel'):
            pd._original_read_excel = pd.read_excel

        def _patched_read_excel(io_path, *args, **kwargs):
            if isinstance(io_path, str):
                io_path = normalizer(io_path)
            return pd._original_read_excel(io_path, *args, **kwargs)
        sandbox_globals['pd'].read_excel = _patched_read_excel

        if not hasattr(pd, '_OriginalExcelFile'):
            pd._OriginalExcelFile = pd.ExcelFile

        class _PatchedExcelFile(pd._OriginalExcelFile):
            def __init__(self, path_or_buffer, *args, **kwargs):
                if isinstance(path_or_buffer, str):
                    path_or_buffer = normalizer(path_or_buffer)
                super().__init__(path_or_buffer, *args, **kwargs)
        sandbox_globals['pd'].ExcelFile = _PatchedExcelFile

        if not hasattr(pd, '_original_read_json'):
            pd._original_read_json = pd.read_json

        def _patched_read_json(path_or_buf, *args, **kwargs):
            if isinstance(path_or_buf, str) and not path_or_buf.startswith(('http://', 'https://', '{', '[')):
                path_or_buf = normalizer(path_or_buf)
            return pd._original_read_json(path_or_buf, *args, **kwargs)
        sandbox_globals['pd'].read_json = _patched_read_json

        if not hasattr(pd, '_original_read_parquet'):
            pd._original_read_parquet = pd.read_parquet

        def _patched_read_parquet(path, *args, **kwargs):
            if isinstance(path, str):
                path = normalizer(path)
            return pd._original_read_parquet(path, *args, **kwargs)
        sandbox_globals['pd'].read_parquet = _patched_read_parquet

        _original_to_csv = pd.DataFrame.to_csv
        def _patched_to_csv(self_df, path_or_buf=None, *args, **kwargs):
            if isinstance(path_or_buf, str):
                path_or_buf = normalizer(path_or_buf)
            return _original_to_csv(self_df, path_or_buf, *args, **kwargs)
        pd.DataFrame.to_csv = _patched_to_csv

        _original_to_excel = pd.DataFrame.to_excel
        def _patched_to_excel(self_df, excel_writer, *args, **kwargs):
            if isinstance(excel_writer, str):
                excel_writer = normalizer(excel_writer)
            return _original_to_excel(self_df, excel_writer, *args, **kwargs)
        pd.DataFrame.to_excel = _patched_to_excel

        _original_to_json = pd.DataFrame.to_json
        def _patched_to_json(self_df, path_or_buf=None, *args, **kwargs):
            if isinstance(path_or_buf, str):
                path_or_buf = normalizer(path_or_buf)
            return _original_to_json(self_df, path_or_buf, *args, **kwargs)
        pd.DataFrame.to_json = _patched_to_json

        _original_to_parquet = pd.DataFrame.to_parquet
        def _patched_to_parquet(self_df, path=None, *args, **kwargs):
            if isinstance(path, str):
                path = normalizer(path)
            return _original_to_parquet(self_df, path, *args, **kwargs)
        pd.DataFrame.to_parquet = _patched_to_parquet

    def _install_chart_hooks(self, sandbox_globals: Dict, chart_capture: ChartCapture):
        plt = sandbox_globals.get('plt')
        if plt is None:
            return
        plt.show = chart_capture.make_show_replacement(plt)
        try:
            import matplotlib.figure
            if not hasattr(matplotlib.figure.Figure, '_original_savefig'):
                matplotlib.figure.Figure._original_savefig = matplotlib.figure.Figure.savefig
            matplotlib.figure.Figure.savefig = chart_capture.make_savefig_wrapper(
                matplotlib.figure.Figure._original_savefig
            )
        except Exception:
            pass

    def _strip_matplotlib_use(self, code: str) -> str:
        def _replace_use(match):
            original = match.group(0).strip()
            return f"# [sandbox] {original} -> already set (Agg backend)"
        return MATPLOTLIB_USE_PATTERN.sub(_replace_use, code)

    def _lazy_import(self, name: str, sandbox_globals: Dict):
        try:
            if name == 'matplotlib' or name == 'matplotlib.pyplot':
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import matplotlib.patches, matplotlib.gridspec
                import matplotlib.ticker, matplotlib.colors, matplotlib.cm
                try: import matplotlib.patheffects
                except ImportError: pass
                try: import matplotlib.image
                except ImportError: pass
                try: import matplotlib.backends.backend_agg
                except ImportError: pass
                try: import matplotlib.backends.backend_pdf
                except ImportError: pass
                sandbox_globals['matplotlib'] = matplotlib
                sandbox_globals['plt'] = plt
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
                try: import plotly.io as pio
                except ImportError: pass
                sandbox_globals['plotly'] = plotly
                sandbox_globals['px'] = px
                sandbox_globals['go'] = go
                return True
            elif name == 'scipy' or name == 'scipy.stats':
                import scipy, scipy.stats
                try: import scipy.optimize
                except ImportError: pass
                try: import scipy.interpolate
                except ImportError: pass
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
                import openpyxl, openpyxl.styles, openpyxl.utils
                try: import openpyxl.chart
                except ImportError: pass
                try: import openpyxl.worksheet.table
                except ImportError: pass
                try: import openpyxl.formatting, openpyxl.formatting.rule
                except ImportError: pass
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
            elif name == 'reportlab':
                import reportlab, reportlab.lib
                import reportlab.lib.pagesizes, reportlab.lib.styles
                import reportlab.lib.units, reportlab.lib.colors, reportlab.lib.enums
                try:
                    import reportlab.platypus
                    import reportlab.platypus.doctemplate, reportlab.platypus.tables
                    import reportlab.platypus.paragraph, reportlab.platypus.spacer
                    import reportlab.platypus.flowables
                except ImportError: pass
                try: import reportlab.pdfgen, reportlab.pdfgen.canvas
                except ImportError: pass
                sandbox_globals['reportlab'] = reportlab
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
            elif name == 'datetime':
                import datetime as _dt_mod
                sandbox_globals['datetime'] = _dt_mod
                sandbox_globals['timedelta'] = _dt_mod.timedelta
                sandbox_globals['timezone'] = _dt_mod.timezone
                sandbox_globals['date'] = _dt_mod.date
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
                import os as os_module
                sandbox_globals['os'] = os_module
                return True
            elif name == 'sys':
                import sys as sys_module
                sandbox_globals['sys'] = sys_module
                return True
            elif name == 'urllib':
                import urllib, urllib.request, urllib.parse, urllib.error
                sandbox_globals['urllib'] = urllib
                return True
            elif name == 'requests':
                try:
                    import requests as requests_module
                    sandbox_globals['requests'] = requests_module
                    return True
                except ImportError:
                    return False
            elif name == 'shutil':
                import shutil
                sandbox_globals['shutil'] = shutil
                return True
            elif name == 'glob':
                import glob
                sandbox_globals['glob'] = glob
                return True
            elif name == 'docx':
                try:
                    import docx
                    sandbox_globals['docx'] = docx
                    try:
                        from docx import Document as _Document
                        sandbox_globals['Document'] = _Document
                    except ImportError:
                        pass
                    return True
                except ImportError:
                    return False
            elif name == 'zipfile':
                import zipfile
                sandbox_globals['zipfile'] = zipfile
                return True
            elif name == 'lxml':
                try:
                    import lxml
                    try: import lxml.etree
                    except ImportError: pass
                    sandbox_globals['lxml'] = lxml
                    return True
                except ImportError:
                    return False
            elif name == 'xml':
                import xml
                try: import xml.etree, xml.etree.ElementTree
                except ImportError: pass
                try: import xml.dom, xml.dom.minidom
                except ImportError: pass
                try: import xml.sax
                except ImportError: pass
                try: import xml.parsers, xml.parsers.expat
                except ImportError: pass
                sandbox_globals['xml'] = xml
                return True
            elif name == 'pkgutil':
                import pkgutil
                sandbox_globals['pkgutil'] = pkgutil
                return True
            elif name == 'importlib':
                import importlib
                try: import importlib.resources
                except ImportError: pass
                try: import importlib.metadata
                except ImportError: pass
                sandbox_globals['importlib'] = importlib
                return True
            elif name == 'warnings':
                import warnings
                sandbox_globals['warnings'] = warnings
                return True
            elif name == 'abc':
                import abc
                sandbox_globals['abc'] = abc
                return True
            elif name == 'enum':
                import enum
                sandbox_globals['enum'] = enum
                return True
            elif name == 'weakref':
                import weakref
                sandbox_globals['weakref'] = weakref
                return True
            elif name == 'io':
                import io as io_module
                sandbox_globals['io'] = io_module
                return True
            elif name == 'copy':
                import copy as copy_module
                sandbox_globals['copy'] = copy_module
                return True
        except ImportError as e:
            logger.warning(f"Failed to import {name}: {e}")
            return False
        return False

    def _preprocess_code(self, code: str, sandbox_globals: Dict) -> str:
        code = self._strip_matplotlib_use(code)
        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                processed_lines.append(line)
                continue

            if stripped.startswith('import ') or stripped.startswith('from '):

                if stripped.startswith('from '):
                    try:
                        parts = stripped.split(' import ', 1)
                        if len(parts) == 2:
                            module_path = parts[0].replace('from ', '').strip()
                            import_names_str = parts[1].strip()
                            base_module = module_path.split('.')[0]

                            if base_module not in sandbox_globals:
                                self._lazy_import(base_module, sandbox_globals)

                            if base_module in sandbox_globals:
                                import_items = [n.strip() for n in import_names_str.split(',')]
                                assignment_lines = []

                                for item in import_items:
                                    item = item.strip()
                                    if not item:
                                        continue
                                    if ' as ' in item:
                                        original, alias = item.split(' as ', 1)
                                        original = original.strip()
                                        alias = alias.strip()
                                    else:
                                        original = item
                                        alias = item

                                    if alias in sandbox_globals and alias == base_module:
                                        continue

                                    assignment_lines.append(f"{alias} = {module_path}.{original}")

                                if assignment_lines:
                                    processed_lines.append(f"# [sandbox] {stripped}")
                                    processed_lines.extend(assignment_lines)
                                    continue
                                else:
                                    processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                    continue
                            else:
                                processed_lines.append(f"# [sandbox] BLOCKED: {stripped}")
                                continue
                        else:
                            module = stripped.split()[1].split('.')[0]
                            if self._lazy_import(module, sandbox_globals):
                                processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                continue
                            elif module in sandbox_globals:
                                processed_lines.append(f"# [sandbox] {stripped} -> available")
                                continue
                            else:
                                processed_lines.append(f"# [sandbox] BLOCKED: {stripped}")
                                continue
                    except Exception as e:
                        logger.warning(f"Failed to parse from-import '{stripped}': {e}")
                        try:
                            module = stripped.split()[1].split('.')[0]
                            if self._lazy_import(module, sandbox_globals):
                                processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                                continue
                            elif module in sandbox_globals:
                                processed_lines.append(f"# [sandbox] {stripped} -> available")
                                continue
                        except Exception:
                            pass
                        processed_lines.append(f"# [sandbox] BLOCKED: {stripped}")
                        continue

                else:
                    module = stripped.split()[1].split('.')[0].split(',')[0]

                    if self._lazy_import(module, sandbox_globals):
                        if ' as ' in stripped:
                            alias = stripped.split(' as ')[-1].strip()
                            if alias not in sandbox_globals:
                                full_module = stripped.split()[1].split(' as ')[0].strip() if ' as ' in stripped.split()[1] else stripped.split()[1]
                                parts = full_module.split('.')
                                obj = sandbox_globals.get(parts[0])
                                if obj and len(parts) > 1:
                                    try:
                                        for part in parts[1:]:
                                            obj = getattr(obj, part)
                                        sandbox_globals[alias] = obj
                                    except AttributeError:
                                        sandbox_globals[alias] = sandbox_globals[module]
                                elif obj:
                                    sandbox_globals[alias] = obj
                        processed_lines.append(f"# [sandbox] {stripped} -> pre-loaded")
                        continue
                    elif module in sandbox_globals:
                        if ' as ' in stripped:
                            alias = stripped.split(' as ')[-1].strip()
                            if alias not in sandbox_globals:
                                full_module = stripped.split()[1]
                                if ' as ' in full_module:
                                    full_module = full_module.split(' as ')[0].strip()
                                parts = full_module.split('.')
                                obj = sandbox_globals.get(parts[0])
                                if obj and len(parts) > 1:
                                    try:
                                        for part in parts[1:]:
                                            obj = getattr(obj, part)
                                        sandbox_globals[alias] = obj
                                    except AttributeError:
                                        sandbox_globals[alias] = sandbox_globals[module]
                                elif obj:
                                    sandbox_globals[alias] = obj
                        processed_lines.append(f"# [sandbox] {stripped} -> available")
                        continue
                    else:
                        processed_lines.append(f"# [sandbox] BLOCKED: {stripped}")
                        continue

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    async def _store_files_in_postgres(self, new_file_paths: List[str],
                                        session_id: str, session_dir: Path) -> List[Dict[str, str]]:
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

                        ext = file_path.suffix.lower()
                        if ext not in STORABLE_EXTENSIONS:
                            continue

                        file_size = file_path.stat().st_size
                        if file_size > max_bytes or file_size == 0:
                            continue

                        file_bytes = file_path.read_bytes()
                        checksum = hashlib.sha256(file_bytes).hexdigest()
                        mime_type = get_mime_type(file_path.name)

                        expires_at = None
                        if ttl_hours > 0:
                            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

                        file_id = uuid.uuid4()
                        sandbox_file = SandboxFile(
                            id=file_id, session_id=session_id,
                            filename=file_path.name, mime_type=mime_type,
                            file_size=file_size, checksum=checksum,
                            content=file_bytes, expires_at=expires_at,
                        )
                        db_session.add(sandbox_file)

                        encoded_filename = quote(file_path.name)
                        url = f"{base_url}/dl/{file_id}/{encoded_filename}" if base_url else f"/dl/{file_id}/{encoded_filename}"

                        if file_size < 1024:
                            size_human = f"{file_size} B"
                        elif file_size < 1024 * 1024:
                            size_human = f"{file_size / 1024:.1f} KB"
                        else:
                            size_human = f"{file_size / (1024 * 1024):.1f} MB"

                        is_image = ext in IMAGE_EXTENSIONS

                        download_info.append({
                            'filename': file_path.name, 'url': url,
                            'size': size_human, 'mime_type': mime_type,
                            'file_id': str(file_id), 'is_image': is_image,
                            'expires_at': expires_at.isoformat() if expires_at else None,
                        })

                    except Exception as e:
                        logger.error(f"Failed to store {rel_path}: {e}")
                        continue

                if download_info:
                    await db_session.commit()

        except Exception as e:
            logger.error(f"File storage failed: {e}", exc_info=True)

        return download_info

    async def execute(self, code: str, session_id: str = "default",
                      timeout: int = None, context: Dict = None) -> ExecutionResult:
        """Execute code in a persistent sandboxed namespace.

        Acquires a per-session lock so concurrent calls to the same
        session_id queue up instead of racing.
        """
        result = ExecutionResult()

        raw_timeout = timeout
        timeout = max(timeout or settings.MAX_EXECUTION_TIME, MIN_EXECUTION_TIMEOUT)
        if raw_timeout and raw_timeout < MIN_EXECUTION_TIMEOUT:
            logger.debug(f"Timeout floor: {raw_timeout}s -> {MIN_EXECUTION_TIMEOUT}s")

        session_dir = self.sandbox_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Serialize concurrent calls to the same session
        exec_lock = kernel_manager.get_exec_lock(session_id)
        exec_lock.acquire()
        try:
            return await self._execute_locked(
                code, session_id, session_dir, timeout, result, context
            )
        finally:
            exec_lock.release()

    async def _execute_locked(self, code: str, session_id: str, session_dir: Path,
                               timeout: int, result: ExecutionResult,
                               context: Dict = None) -> ExecutionResult:
        """Execute code while holding the per-session lock."""

        logger.info(f"execute: session={session_id}, timeout={timeout}s")

        try:
            existing_globals = kernel_manager.get_existing(session_id)
            if existing_globals is not None:
                sandbox_globals = existing_globals
                user_vars = [k for k in sandbox_globals if not k.startswith('_') and k not in (
                    'pd','pandas','np','numpy','json','csv','math','statistics','datetime',
                    'collections','itertools','functools','re','io','copy','hashlib','base64',
                    'Path','SANDBOX_DIR','RESULT','plt','matplotlib','sns','seaborn',
                    'Dict','List','Optional','Tuple','Set','Any','timedelta','timezone','date',
                    'Decimal','Fraction','dataclass','field','asdict',
                )]
                logger.error(f"KERNEL_DIAG: REUSED session={session_id}, "
                             f"user_vars={user_vars[:10]}, total_keys={len(sandbox_globals)}")
            else:
                fresh_globals = self._build_safe_globals(session_dir)
                sandbox_globals = kernel_manager.get_or_create(
                    session_id=session_id,
                    sandbox_globals=fresh_globals,
                    session_dir=session_dir,
                )
                logger.error(f"KERNEL_DIAG: CREATED session={session_id}, "
                             f"total_keys={len(sandbox_globals)}, "
                             f"active_kernels={kernel_manager.active_count}")
        except Exception as e:
            result.success = False
            result.error_message = f"Kernel initialization failed: {e}"
            result.error_traceback = traceback.format_exc()
            return result

        sandbox_globals['RESULT'] = None
        if context:
            sandbox_globals.update(context)

        sandbox_globals['_normalize_path'] = self._make_path_normalizer(session_dir)
        self._install_pandas_path_hooks(sandbox_globals, session_dir)

        try:
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

        files_before = set()
        if session_dir.exists():
            try:
                files_before = set(str(p) for p in session_dir.rglob('*') if p.is_file())
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
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            logger.info(f"done: success={result.success}, {result.execution_time_ms}ms, "
                        f"charts={len(chart_capture.captured_charts)}")

            if result.error_message:
                logger.error(f"error: {result.error_message}")

            if HAS_RESOURCE:
                try:
                    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
                    result.memory_used_mb = usage.ru_maxrss / 1024
                except Exception:
                    result.memory_used_mb = 0.0

            if session_dir.exists():
                try:
                    files_after = set(str(p) for p in session_dir.rglob('*') if p.is_file())
                    new_files = files_after - files_before
                    result.files_created = [str(Path(f).relative_to(session_dir)) for f in new_files]
                except Exception:
                    pass

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
                if non_image_downloads:
                    url_lines = ["\n\nGenerated files ready for download:"]
                    for info in non_image_downloads:
                        url_lines.append(f"\n[{info['filename']} ({info['size']})]({info['url']})")
                    result.stdout = result.stdout + '\n'.join(url_lines)

                if result.inline_images:
                    img_lines = ["\n\nGenerated charts:"]
                    for img in result.inline_images:
                        img_lines.append(f"\n![{img['alt_text']}]({img['url']})")
                    result.stdout = result.stdout + '\n'.join(img_lines)

            except Exception as e:
                logger.error(f"File storage failed (non-fatal): {e}", exc_info=True)

        return result


executor = SandboxExecutor()
