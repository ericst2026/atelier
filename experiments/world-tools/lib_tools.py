"""Three tools, the syntax the model writes, and the loop that runs them."""
import ast
import operator
import re
from typing import Any, Callable

CALL = re.compile(r"([a-z_]+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
RESULT_OPEN, RESULT_CLOSE = "[=", "]"
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow, ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calc(expression: str) -> str:
    """Arithmetic only — parsed, never exec'd, so a student cannot smuggle code through a tool call."""
    value = _eval(ast.parse(expression.strip(), mode="eval"))
    if isinstance(value, float) and abs(value - round(value)) < 1e-9:
        value = int(round(value))
    return str(round(value, 4) if isinstance(value, float) else value)


def count(items: str) -> str:
    """count(3, 8, 1) → the sum; count of a comma-separated list of numbers."""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", items)]
    total = sum(nums)
    return str(int(total) if abs(total - round(total)) < 1e-9 else round(total, 4))


def make_lookup(facts: dict[str, str]) -> Callable[[str], str]:
    def lookup(key: str) -> str:
        key = key.strip().strip("'\"")
        if key in facts:
            return facts[key]
        for k, v in facts.items():
            if key.lower() in k.lower():
                return v
        return "not found"
    return lookup


def render(name: str, args: str, result: str) -> str:
    return f"{name}({args}){RESULT_OPEN}{result}{RESULT_CLOSE}"


def find_call(text: str, tools: dict[str, Callable]) -> tuple[str, str] | None:
    """The first call to a known tool that has not already been answered."""
    for m in CALL.finditer(text):
        name, args = m.group(1), m.group(2)
        if name in tools and text[m.end() : m.end() + len(RESULT_OPEN)] != RESULT_OPEN:
            return name, args
    return None


EXPR = re.compile(r"[\d(][\d\s.+\-*/()]*[\d)]")


def expressions(text: str) -> list[str]:
    """Arithmetic written in a line of text: digits joined by at least one operator."""
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    out = []
    for m in EXPR.finditer(text):
        e = m.group(0).strip()
        if any(op in e for op in "+-*/") and "**" not in e and re.search(r"\d\s*[+\-*/]\s*[\d(]", e) and e.count("(") == e.count(")"):
            out.append(e)
    return out


def prepared_call(task: dict[str, Any], enabled: set[str], grade: Callable[[str], bool]) -> tuple[str, str, str] | None:
    """For a question from a prepared dataset (no generator family to go by): a tool call
    whose result is itself the right answer, or None. Arithmetic is looked for in the
    left-hand sides of the row's steps, then in the question; numbers to add up in the
    question. `grade(result)` says whether a result is the answer."""
    if "calc" in enabled:
        lines = [str(s).split("=")[0] for s in task.get("steps") or [] if "=" in str(s)]
        for expr in [e for line in lines for e in expressions(line)] + expressions(task["prompt"]):
            try:
                result = calc(expr)
            except Exception:
                continue
            if grade(result):
                return "calc", expr, result
    if "count" in enabled:
        numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.]\d)", task["prompt"])
        if len(numbers) >= 2:
            args = ", ".join(numbers)
            try:
                result = count(args)
            except Exception:
                result = None
            if result is not None and grade(result):
                return "count", args, result
    return None


def run_call(tools: dict[str, Callable], name: str, args: str) -> tuple[str, bool]:
    try:
        return str(tools[name](args)), True
    except Exception as exc:
        return f"error: {type(exc).__name__}", False
