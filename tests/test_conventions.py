# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = sorted(
    path
    for directory in ("nalp", "examples", "tests", "docs")
    for path in _ROOT.joinpath(directory).rglob("*.py")
    if not {"__pycache__", "_build", "generated"}.intersection(path.relative_to(_ROOT).parts)
)
_LIBRARY_SOURCES = [path for path in _SOURCES if path.relative_to(_ROOT).parts[0] == "nalp"]


@pytest.mark.parametrize("path", _SOURCES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_python_file_has_header_absolute_top_level_imports_and_concise_comments(path):
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    assert lines[:3] == [
        "# Copyright (c) 2019-2026 Gustavo de Rosa.",
        "# Licensed under the Apache License, Version 2.0.",
        "",
    ]

    top_level_imports = {id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert id(node) in top_level_imports
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            if node.module == "typing":
                assert not {
                    "Optional",
                    "Union",
                    "List",
                    "Dict",
                    "Tuple",
                    "Set",
                    "Callable",
                    "Iterable",
                    "Iterator",
                    "Mapping",
                    "Sequence",
                }.intersection(alias.name for alias in node.names)
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None

    previous_line = 0
    comment_run = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or token.start[0] <= 2:
            continue
        if token.string.startswith(("# noqa", "# type:", "# pragma:", "# fmt:")):
            continue

        assert len(lines[token.start[0] - 1]) <= 120
        assert not token.string.rstrip().endswith(".")
        assert not re.fullmatch(r"#[\s=*_#-]+", token.string)
        if not lines[token.start[0] - 1][: token.start[1]].strip():
            comment_run = comment_run + 1 if token.start[0] == previous_line + 1 else 1
            previous_line = token.start[0]
            assert comment_run <= 3


@pytest.mark.parametrize("path", _LIBRARY_SOURCES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_library_docstrings_describe_public_apis_and_omit_private_framework_hooks(path):
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parent = parents[node]
        public_scope = isinstance(parent, (ast.Module, ast.ClassDef))
        if isinstance(parent, ast.ClassDef) and parent.name.startswith("_"):
            public_scope = False
        framework_hook = isinstance(parent, ast.ClassDef) and node.name in {
            "call",
            "build",
            "get_config",
            "get_initial_state",
        }
        private = node.name.startswith("_") and node.name != "__init__"
        docstring = ast.get_docstring(node)

        if private or framework_hook or not public_scope:
            assert docstring is None
            continue

        assert docstring is not None
        assert docstring.splitlines()[0].endswith(".")

        doc_node = node.body[0]
        assert all(len(line) <= 120 for line in lines[doc_node.lineno - 1 : doc_node.end_lineno])
        if doc_node.end_lineno > doc_node.lineno:
            assert not lines[doc_node.end_lineno - 2].strip()
        if len(node.body) > 1:
            assert not lines[doc_node.end_lineno].strip()
            assert lines[doc_node.end_lineno + 1].strip()

        if isinstance(node, ast.ClassDef):
            fields = [
                member.target.id
                for member in node.body
                if isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
                and not member.target.id.startswith("_")
            ]
            has_constructor = any(
                isinstance(member, ast.FunctionDef) and member.name == "__init__" for member in node.body
            )
            if fields and not has_constructor:
                assert "Attributes:" in docstring
                for field in fields:
                    if not field.startswith("_"):
                        assert re.search(rf"^\s+{re.escape(field)}:", docstring, re.MULTILINE)
            else:
                assert len(docstring.splitlines()) == 1
            continue

        parameters = {argument.arg for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]} - {
            "self",
            "cls",
        }
        parameters.update(argument.arg for argument in (node.args.vararg, node.args.kwarg) if argument is not None)
        if parameters:
            assert "Args:" in docstring
            for parameter in parameters:
                assert re.search(rf"^\s+\**{re.escape(parameter)}:", docstring, re.MULTILINE)

        section = None
        for line in docstring.splitlines()[1:]:
            if line and not line.startswith(" "):
                section = line.removesuffix(":") if line.endswith(":") else None
                continue
            if line.strip() and section in {"Args", "Returns", "Raises", "Attributes"}:
                assert ";" not in line
                assert not re.search(r"\bdefaults?\s+to\b", line, re.IGNORECASE)
                assert len(line) - len(line.lstrip()) == 4
                if section in {"Args", "Raises", "Attributes"}:
                    assert re.match(r"    \*{0,2}[\w.]+:", line)


@pytest.mark.parametrize("path", _LIBRARY_SOURCES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_library_validation_and_error_messages_follow_conventions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        assert not isinstance(node, ast.Assert)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "print"
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or not node.exc.args:
            continue

        message = node.exc.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            prefix = suffix = message.value
        elif isinstance(message, ast.JoinedStr):
            assert isinstance(message.values[0], ast.Constant)
            assert isinstance(message.values[-1], ast.Constant)
            prefix = message.values[0].value
            suffix = message.values[-1].value
        else:
            continue

        assert prefix.startswith("`")
        assert suffix.endswith(".")
