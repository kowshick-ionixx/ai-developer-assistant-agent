"""Safe mathematical expression parser using AST."""

import ast
import math
import operator

from .basic import (
    factorial,
    square_root,
)
from .scientific import (
    PI,
    E,
    acos_func,
    asin_func,
    atan_func,
    cos_func,
    cosh_func,
    exp_func,
    ln_func,
    log10_func,
    sin_func,
    sinh_func,
    tan_func,
    tanh_func,
)

SUPPORTED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SUPPORTED_FUNCTIONS = {
    "sin": sin_func,
    "cos": cos_func,
    "tan": tan_func,
    "asin": asin_func,
    "acos": acos_func,
    "atan": atan_func,
    "sinh": sinh_func,
    "cosh": cosh_func,
    "tanh": tanh_func,
    "log10": log10_func,
    "ln": ln_func,
    "log": ln_func,
    "exp": exp_func,
    "sqrt": square_root,
    "fact": factorial,
    "factorial": factorial,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}

SUPPORTED_CONSTANTS = {
    "pi": PI,
    "e": E,
}


class SafeEvalVisitor(ast.NodeVisitor):
    def __init__(self, mode="rad"):
        self.mode = mode

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in SUPPORTED_OPERATORS:
                left = self.visit(node.left)
                right = self.visit(node.right)
                try:
                    return SUPPORTED_OPERATORS[op_type](left, right)
                except ZeroDivisionError:
                    raise ValueError("Division by zero")
                except Exception as e:
                    raise ValueError(f"Calculation error: {e}")
            raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in SUPPORTED_OPERATORS:
                operand = self.visit(node.operand)
                return SUPPORTED_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Name):
            name = node.id.lower()
            if name in SUPPORTED_CONSTANTS:
                return SUPPORTED_CONSTANTS[name]
            raise ValueError(f"Unsupported variable or constant: {node.id}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.lower()
                if func_name in SUPPORTED_FUNCTIONS:
                    args = [self.visit(arg) for arg in node.args]
                    func = SUPPORTED_FUNCTIONS[func_name]
                    try:
                        if func_name in ["sin", "cos", "tan", "asin", "acos", "atan"]:
                            return func(*args, mode=self.mode)
                        else:
                            return func(*args)
                    except TypeError as te:
                        raise ValueError(f"Invalid arguments for {func_name}: {te}")
                    except Exception as e:
                        raise ValueError(f"{e}")
                raise ValueError(f"Unsupported function: {node.func.id}")
            raise ValueError("Unsupported function call structure")
        else:
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


def evaluate_expression(expr: str, mode: str = "rad") -> float:
    if not expr or not expr.strip():
        raise ValueError("Empty expression")
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        visitor = SafeEvalVisitor(mode=mode)
        result = visitor.visit(tree)
        if not math.isfinite(result):
            raise ValueError("Result is not a finite number")
        return result
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Evaluation error: {e}")
