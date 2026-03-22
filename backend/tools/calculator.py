import math
import re

ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
ALLOWED_NAMES.update({
    "abs": abs, "round": round, "int": int, "float": float,
    "sum": sum, "min": min, "max": max, "pow": pow,
})

def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Supports: basic arithmetic, math.* functions, constants (pi, e, tau).
    """
    # Clean up expression
    expr = expression.strip()
    expr = expr.replace("^", "**")  # support ^ as power
    expr = re.sub(r'\bpi\b', 'pi', expr)
    expr = re.sub(r'\be\b', 'e', expr)

    # Block anything non-math
    if re.search(r'[a-zA-Z_]\w*\s*\(', expr):
        allowed_fns = set(ALLOWED_NAMES.keys())
        used_fns = re.findall(r'([a-zA-Z_]\w*)\s*\(', expr)
        for fn in used_fns:
            if fn not in allowed_fns:
                return {
                    "expression": expression,
                    "error": f"Function '{fn}' is not allowed.",
                    "success": False
                }

    try:
        result = eval(expr, {"__builtins__": {}}, ALLOWED_NAMES)
        if isinstance(result, float):
            if math.isnan(result):
                return {"expression": expression, "error": "Result is NaN", "success": False}
            if math.isinf(result):
                return {"expression": expression, "error": "Result is infinite", "success": False}
            result = round(result, 10)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero", "success": False}
    except SyntaxError as e:
        return {"expression": expression, "error": f"Syntax error: {e}", "success": False}
    except Exception as e:
        return {"expression": expression, "error": str(e), "success": False}
