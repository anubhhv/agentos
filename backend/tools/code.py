import sys
import ast
import math
import json
import traceback
import contextlib
import io
import signal
from typing import Any

# Whitelist of safe built-ins
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "ascii": ascii,
    "bin": bin, "bool": bool, "bytearray": bytearray, "bytes": bytes,
    "callable": callable, "chr": chr, "complex": complex,
    "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
    "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "setattr": setattr,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "vars": vars, "zip": zip,
    "__build_class__": __build_class__, "__name__": "__main__",
    "True": True, "False": False, "None": None,
    # Safe exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError, "AttributeError": AttributeError,
}

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "math": math,
    "json": json,
}


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (10s limit)")


def run_python(code: str, timeout: int = 10) -> dict:
    """
    Execute Python code in a restricted sandbox.
    Returns stdout, stderr, return value, and any errors.
    """
    # Basic static analysis — block dangerous patterns
    BLOCKED = [
        "import os", "import sys", "import subprocess", "import socket",
        "import shutil", "import pathlib", "open(", "exec(", "eval(",
        "__import__", "importlib", "builtins", "globals()", "locals()",
        "compile(", "getattr(", "setattr(", "delattr(",
    ]
    for blocked in BLOCKED:
        if blocked in code:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Blocked: '{blocked}' is not allowed in sandbox.",
                "error": "SecurityError",
                "return_value": None
            }

    # Parse AST to catch import statements
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                allowed = {"math", "json", "random", "datetime", "itertools",
                           "functools", "collections", "string", "re", "statistics",
                           "decimal", "fractions", "heapq", "bisect"}
                for name in names:
                    base = name.split(".")[0]
                    if base not in allowed:
                        return {
                            "success": False,
                            "stdout": "",
                            "stderr": f"Import '{name}' is not allowed. Allowed: {sorted(allowed)}",
                            "error": "SecurityError",
                            "return_value": None
                        }
    except SyntaxError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"SyntaxError: {e}",
            "error": "SyntaxError",
            "return_value": None
        }

    # Build safe globals with allowed imports
    import random, datetime, itertools, functools, collections, string, re, statistics
    import decimal, fractions, heapq, bisect

    safe_globals = dict(SAFE_GLOBALS)
    safe_globals.update({
        "random": random, "datetime": datetime, "itertools": itertools,
        "functools": functools, "collections": collections, "string": string,
        "re": re, "statistics": statistics, "decimal": decimal,
        "fractions": fractions, "heapq": heapq, "bisect": bisect,
    })

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    return_value = None

    try:
        # Set up timeout on Unix systems
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            local_vars = {}
            exec(compile(code, "<sandbox>", "exec"), safe_globals, local_vars)
            # Try to get the last expression value
            lines = code.strip().splitlines()
            if lines:
                try:
                    return_value = eval(lines[-1], safe_globals, local_vars)
                except Exception:
                    pass

        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": None,
            "return_value": repr(return_value) if return_value is not None else None
        }

    except TimeoutError as e:
        return {"success": False, "stdout": stdout_capture.getvalue(),
                "stderr": str(e), "error": "TimeoutError", "return_value": None}
    except Exception as e:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": traceback.format_exc(),
            "error": type(e).__name__,
            "return_value": None
        }


def evaluate_expression(expr: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        # Only allow math expressions
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return {"expression": expr, "result": result, "success": True}
    except Exception as e:
        return {"expression": expr, "error": str(e), "success": False}
