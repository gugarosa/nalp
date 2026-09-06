# NALP conventions

NALP adopts the code-style conventions from
[cpmux](https://github.com/gugarosa/cpmux/blob/main/CONVENTIONS.md), originally derived from phitrain.
These rules apply to library code, examples, tests, and documentation Python files.

## Compatibility and scope

- Preserve public import paths, constructor parameters, return structures, and supported model behavior.
- Keep NALP's Apache-2.0 license and declared Python 3.11-3.13 support. The modern syntax below is compatible with
  Python 3.11; a style change must not silently raise the supported interpreter floor.
- Use `nalp` absolute imports and the existing `nalp.utils.logging` API, not cpmux package paths.
- Retain NALP's public package exports. cpmux's empty package initializers and application-specific architecture
  are not conventions to transplant into this library.
- Preserve learned-weight layouts and documented state behavior. Checkpoint-format changes are separate API work.

## Python and imports

- Use `X | None`, never `Optional[X]`, and builtin generics such as `dict[str, Any]` and `list[str]` (R2).
- Import abstract collection types such as `Callable`, `Iterable`, and `Sequence` from `collections.abc`.
- Import only types without builtin equivalents, such as `Any`, `Literal`, and `Annotated`, from `typing`.
- Keep imports at module scope and absolute, ordered as standard library, third-party packages, then NALP,
  with a blank line between groups.
- Begin every Python file with this header:

  ```python
  # Copyright (c) 2019-2026 Gustavo de Rosa.
  # Licensed under the Apache License, Version 2.0.
  ```

## Docstrings

- Use Google-style docstrings for public functions, classes, and explicitly implemented public constructors (R3, R13).
- Start with a single-sentence summary. A regular class has a one-line summary; constructor `Args:` belong on
  `__init__`, not on the class.
- Keep each `Args:`, `Returns:`, and `Raises:` entry on one line. Do not add semicolons or `defaults to <X>` tails.
- Keep summary-only docstrings on one line, as in cpmux's Black-formatted code. A multiline docstring has one blank
  line before its closing `"""`. Leave one blank line after a docstring before executable code or class fields.
- Private helpers and framework-dispatched hooks have no docstrings. For Keras these include `call`, `build`,
  `get_config`, and `get_initial_state`. Document their NALP-specific tensor and state contracts in constructor
  documentation and the API contract guide instead.
- Preserve scientific references and explanations of shapes, dtypes, logits/probabilities, mutation, persistence,
  resource ownership, and failures. A shorter docstring is not an excuse to remove a supported contract.
- Put scientific references in constructor documentation or the API guide when the class summary alone is insufficient.
- Data classes without an explicit constructor document every public field in `Attributes:`, one line per field.
- Pytest tests and fixtures are framework entrypoints, not public library APIs; they have no docstrings or type hints.

## Errors and logging

- Validate with `if` and a specific exception, never production `assert`. Test assertions remain ordinary assertions.
- Do not use bare `except:` or broad catches that conceal failures.
- Format raised messages as ``"`name` <verb phrase>, but got <value>."`` when a value is useful (R1).
  End messages with a period and use `is None` or `is True` wording where relevant.
- Use `get_logger(__name__)` from `nalp.utils.logging` when library logging is needed.
- Do not use `print()` in library code. Examples may present their results with `print()`; do not add Rich or Typer
  merely to copy cpmux's application presentation layer.
- Diagnostic warnings and errors identify a backticked offender and end with a period, for example
  ``logger.warning(f"`name={value}` could not be loaded.")`` (R14).
- Keep informational and debug logging plain. Do not log and re-raise the same failure unnecessarily.

## Readability and abstraction

- Comments explain why, not what. Prefer no comment or one line, with a three-line maximum, no banners,
  and no trailing period (R8). License headers and required tool directives are exempt.
- Insert one blank line at meaningful phase transitions in function bodies of at least 12 lines (R11).
  Separate validation, preparation, computation, and publication where those phases exist.
- Inline first; extract a helper, constant, or parameter only when a second call site establishes shared
  responsibility (R16). Do not delete a public API or framework extension hook because internal usage is sparse.
- Keep algorithm-specific losses, rewards, and update ordering explicit. Do not introduce a generic trainer,
  registry, configuration hierarchy, or additional package layer solely for stylistic uniformity.
- Use double quotes and keep readable prose within 120 characters (R9).

## Tooling and review

Black, isort with the Black profile, and Flake8 use a 120-character line length. Flake8's native configuration
is shared by direct CLI invocation, editors, and pre-commit hooks.

Raw corpora under `data/` are not source-formatting targets. Their line endings and trailing whitespace are
training input, so whitespace-rewriting hooks exclude that directory.

```bash
uv run pre-commit run --all-files
uv run pytest
uv run --group docs python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Review the rendered API, not only the documentation build result. Constructor parameters, public state,
and scientific references must remain discoverable. Preserve the established behavior and artifact checks
when changing names, formatting, or documentation.
