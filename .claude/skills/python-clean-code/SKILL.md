---
name: python-clean-code
description: Python clean code review skill. Use this skill whenever reviewing, auditing, or finalizing Python code changes — including PR reviews, pre-commit checks, refactoring passes, or when the user asks to "review", "clean up", or "check" Python code. Trigger even if the user just says "review my changes" without specifying Python explicitly, as long as Python files are involved.
---

# Python Clean Code Review

When triggered, systematically check **every** item below against the code under review. Report violations grouped by file, with line references and concrete fixes.



## 1. Import Hygiene
Imports go on top of each module (except inside `if __name__=='__main__') and have the following structure
1. standard library imports
(empty line)
2. dependency imports
(empty line)
3. local imports.

Sort them alphabetically according to the package/module name

## 2. Naming

- **Functions**: verb-first, intent-describing (`parse_config`, `validate_token`). Never generic (`process`, `handle`, `do_thing`).
- **Variables**: descriptive nouns matching their content (`user_count`, not `n` or `tmp`). Loop counters (`i`, `j`) and well-known short names (`df`, `db`) are acceptable.
- **Constants**: `UPPER_SNAKE_CASE`.
- **Classes**: `PascalCase`. No suffix like `Manager` or `Helper` unless the role is genuinely that.
- **Boolean vars/params**: prefix with `is_`, `has_`, `should_`, `can_`.

## 3. Function Design

- **Atomic**: one responsibility per function. If the docstring needs "and", split it.
- **Keyword-only arguments**: all parameters after `self`/`cls` must be keyword-only.
- Do NOT use bare `*` or `**` splat operators in signatures (DO NOT: `my_func(name: str, *, age: int)` or `my_func(name: str, **, age: int)`)

## 4. Docstrings

Short, single-purpose summary. No `Returns:` section — the function name + return type annotation must make this obvious. No restating of parameter types already in the signature.

## 5. String Formatting

Use f-strings exclusively. Flag and replace all `%`-style and `.format()` usage.

## 6. Exception Handling

- Never bare `except:` or `except Exception:` without re-raise or logging.
- Catch the most specific exception type possible.
- Always log exceptions with `exc_info` or `logger.exception()`.
- Use context in the log message — what operation failed and with what input.
- No silenced exceptions (empty `except` blocks that `pass`).

## 7. Dead & Deprecated Code

Flag and remove:
- Unused functions, classes, variables, or imports.
- `TODO`/`FIXME` markers older than the current change scope — note them for the author.
- Deprecated standard library usage (e.g., `typing.Optional` → `X | None` on Python 3.10+, `os.path` → `pathlib`).

## 8. PEP Compliance

- **PEP 8**: line length ≤ 120 (strict). Consistent within project.
- **PEP 257**: docstring conventions (see §4).
- **PEP 484 / 585**: type hints on all public function signatures. Use built-in generics (`list[str]`, `dict[str, int]`) on Python 3.9+.
- **PEP 20** (Zen): flag unnecessarily clever code. Explicit over implicit. Simple over complex.

## 9. Review Output Format

Structure your review as:

```
### <filename>

**Line X**: <category> — <what's wrong>
→ <concrete fix or replacement code>

**Line Y**: ...
```
