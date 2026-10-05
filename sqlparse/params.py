#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Parameter placeholder inventory for SQL statements.

This module scans parsed SQL for named and positional parameter
placeholders and records, for every occurrence, the original source
span, the statement number, and the occurrence order.  The scan is
purely lexical -- SQL is never executed.

The following regions are excluded from the scan, so placeholders
inside them are never recorded:

* string literals (``'...'``),
* comments (``-- ...`` and ``/* ... */``),
* quoted identifiers (``"..."``, ```...``` and ``[...]``),
* PostgreSQL dollar-quoted literals (``$tag$...$tag$``).

The interpretation of the question mark depends on an explicit
*dialect*: in PostgreSQL ``?`` (and ``?|`` / ``?&``) is a JSON
operator, in MySQL/SQLite it is a positional placeholder, and with an
unknown dialect it is marked as unknown.  Same-named placeholders keep
a separate record for every occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import sqlparse
from sqlparse import tokens as T

__all__ = ["Parameter", "ParameterInventory", "scan"]

# ---------------------------------------------------------------------------
# Dialect rules
# ---------------------------------------------------------------------------
# Each rule set describes which placeholder styles a dialect understands:
#
# * ``named_prefixes``: prefix characters that start a named placeholder
#   (``:name``, ``@name``, ``$name``).
# * ``positional_prefixes``: prefix characters that start a numbered
#   placeholder (``$1``, ``$2``).
# * ``qmark``: interpretation of a bare ``?`` -- ``"placeholder"``,
#   ``"operator"`` (JSON operator) or ``"unknown"``.

_DIALECT_RULES = {
    "postgresql": {
        "named_prefixes": (":",),
        "positional_prefixes": ("$",),
        "qmark": "operator",
    },
    "mysql": {
        "named_prefixes": (":", "@"),
        "positional_prefixes": (),
        "qmark": "placeholder",
    },
    "sqlite": {
        "named_prefixes": (":", "@", "$"),
        "positional_prefixes": (),
        "qmark": "placeholder",
    },
    "oracle": {
        "named_prefixes": (":",),
        "positional_prefixes": (),
        "qmark": "unknown",
    },
    "unknown": {
        "named_prefixes": (":",),
        "positional_prefixes": (),
        "qmark": "unknown",
    },
}

#: Dialect used when no explicit dialect is given (or ``None`` is passed).
DEFAULT_DIALECT = "unknown"

_VALID_DIALECTS = tuple(_DIALECT_RULES)


def _rules_for(dialect):
    if dialect is None:
        dialect = DEFAULT_DIALECT
    if dialect not in _DIALECT_RULES and dialect != DEFAULT_DIALECT:
        raise ValueError(
            f"unknown dialect {dialect!r}; expected one of {_VALID_DIALECTS}"
        )
    return _DIALECT_RULES.get(dialect, _DIALECT_RULES[DEFAULT_DIALECT])

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    """A single parameter placeholder occurrence.

    Attributes:
        name: The parameter name for named placeholders, or ``None`` for
            positional placeholders and JSON operators.
        style: ``"named"`` or ``"positional"``.
        kind: ``"placeholder"`` for real placeholders, ``"json_operator"``
            for the PostgreSQL JSON question-mark operator, or
            ``"unknown"`` for constructs the dialect cannot classify.
        text: The original source text of the occurrence, e.g. ``":id"``,
            ``"?"``, ``"$1"`` or ``"?|"``.
        start: Zero-based offset of the first character in the source.
        end: Zero-based offset just past the last character.
        statement: One-based number of the statement this occurrence
            belongs to.
        index: Zero-based occurrence order within the statement.
        order: Zero-based occurrence order across all statements.
        dialect: The dialect used for classification.
    """

    name: str | None
    style: str
    kind: str
    text: str
    start: int
    end: int
    statement: int
    index: int
    order: int
    dialect: str

    @property
    def span(self) -> tuple[int, int]:
        """The ``(start, end)`` source span of the occurrence."""
        return (self.start, self.end)

    def to_dict(self) -> dict:
        """Return the occurrence as a plain dictionary."""
        return {
            "name": self.name,
            "style": self.style,
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "statement": self.statement,
            "index": self.index,
            "order": self.order,
            "dialect": self.dialect,
        }

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
class ParameterInventory:
    """Inventory of parameter placeholders found in a SQL text.

    Parameters are recorded in source order.  Use the filtered views
    :attr:`placeholders`, :attr:`named`, :attr:`positional`,
    :attr:`json_operators` and :attr:`unknowns` to access specific
    categories.
    """

    def __init__(self, sql: str, dialect: str | None = None):
        self.sql = sql
        self.dialect = dialect if dialect is not None else DEFAULT_DIALECT
        self._rules = _rules_for(self.dialect)
        self._parameters = tuple(self._scan(sql))

    # -- scanning -----------------------------------------------------------
    def _scan(self, sql: str):
        statements = sqlparse.parse(sql)
        # Build a flat list of (statement_number, token, start, end) tuples
        # with global source offsets.  Flattened tokens reconstruct the
        # original text exactly, so offsets are accumulated token by token.
        flat = []
        offset = 0
        for stmt_no, stmt in enumerate(statements, start=1):
            for tok in stmt.flatten():
                start = offset
                end = offset + len(tok.value)
                flat.append((stmt_no, tok, start, end))
                offset = end

        stmt_counters = {}
        global_order = 0
        for i, (stmt_no, tok, start, end) in enumerate(flat):
            nxt = flat[i + 1] if i + 1 < len(flat) else None
            record = self._classify(tok, start, end, nxt)
            if record is None:
                continue
            stmt_index = stmt_counters.get(stmt_no, 0)
            stmt_counters[stmt_no] = stmt_index + 1
            yield Parameter(
                name=record["name"],
                style=record["style"],
                kind=record["kind"],
                text=record["text"],
                start=start,
                end=record["end"],
                statement=stmt_no,
                index=stmt_index,
                order=global_order,
                dialect=self.dialect,
            )
            global_order += 1

    def _classify(self, tok, start, end, nxt):
        """Return a record dict for *tok*, or ``None`` if it is excluded."""
        ttype = tok.ttype
        value = tok.value

        # --- excluded regions ---------------------------------------------
        if ttype in T.Comment:
            return None
        if ttype in T.String:
            # String literals and quoted identifiers ("...", Symbol).
            return None
        if ttype is T.Literal:
            # PostgreSQL dollar-quoted literal.
            return None
        if ttype is T.Name.Placeholder:
            return self._classify_placeholder(value, start, end, nxt)
        if ttype in T.Name:
            # Quoted identifiers (`...`, [...]) and @-variables.
            if value.startswith("`") or value.startswith("["):
                return None
            if value.startswith("@"):
                return self._classify_at_name(value, start, end)
            return None
        return None

    def _classify_placeholder(self, value, start, end, nxt):
        if value == "?":
            qmark = self._rules["qmark"]
            if qmark == "operator":
                # In PostgreSQL ?| and ?& are JSON operators too; merge
                # when the next token is a contiguous | or & operator.
                text = value
                rec_end = end
                if (
                    nxt is not None
                    and nxt[1].ttype is T.Operator
                    and nxt[1].value in ("|", "&")
                    and nxt[2] == end
                ):
                    text = value + nxt[1].value
                    rec_end = nxt[3]
                return {
                    "name": None,
                    "style": "positional",
                    "kind": "json_operator",
                    "text": text,
                    "end": rec_end,
                }
            if qmark == "placeholder":
                return {
                    "name": None,
                    "style": "positional",
                    "kind": "placeholder",
                    "text": value,
                    "end": end,
                }
            return {
                "name": None,
                "style": "positional",
                "kind": "unknown",
                "text": value,
                "end": end,
            }

        if value.startswith(":"):
            return {
                "name": value[1:],
                "style": "named",
                "kind": "placeholder",
                "text": value,
                "end": end,
            }

        if value.startswith("$"):
            rest = value[1:]
            if "$" in self._rules["positional_prefixes"] and rest.isdigit():
                return {
                    "name": rest,
                    "style": "positional",
                    "kind": "placeholder",
                    "text": value,
                    "end": end,
                }
            if "$" in self._rules["named_prefixes"]:
                return {
                    "name": rest,
                    "style": "named",
                    "kind": "placeholder",
                    "text": value,
                    "end": end,
                }
            return {
                "name": rest,
                "style": "positional",
                "kind": "unknown",
                "text": value,
                "end": end,
            }

        if value.startswith("%"):
            if value == "%s":
                return {
                    "name": None,
                    "style": "positional",
                    "kind": "placeholder",
                    "text": value,
                    "end": end,
                }
            if value.startswith("%(") and value.endswith(")s"):
                return {
                    "name": value[2:-2],
                    "style": "named",
                    "kind": "placeholder",
                    "text": value,
                    "end": end,
                }
            return None

        return None

    def _classify_at_name(self, value, start, end):
        name = value[1:]  # strip '@'
        if "@" in self._rules["named_prefixes"]:
            return {
                "name": name,
                "style": "named",
                "kind": "placeholder",
                "text": value,
                "end": end,
            }
        return {
            "name": name,
            "style": "named",
            "kind": "unknown",
            "text": value,
            "end": end,
        }

    # -- views --------------------------------------------------------------
    @property
    def parameters(self):
        """All recorded occurrences in source order."""
        return self._parameters

    @property
    def placeholders(self):
        """Real placeholders (``kind == "placeholder"``)."""
        return tuple(p for p in self._parameters if p.kind == "placeholder")

    @property
    def named(self):
        """Named placeholders in source order."""
        return tuple(p for p in self.placeholders if p.style == "named")

    @property
    def positional(self):
        """Positional placeholders in source order."""
        return tuple(p for p in self.placeholders if p.style == "positional")

    @property
    def json_operators(self):
        """PostgreSQL JSON question-mark operators."""
        return tuple(p for p in self._parameters if p.kind == "json_operator")

    @property
    def unknowns(self):
        """Occurrences the dialect cannot classify."""
        return tuple(p for p in self._parameters if p.kind == "unknown")

    @property
    def statement_count(self):
        """Number of SQL statements scanned."""
        return len(sqlparse.parse(self.sql))

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __repr__(self):
        return (
            f"<ParameterInventory dialect={self.dialect!r} "
            f"placeholders={len(self.placeholders)} "
            f"json_operators={len(self.json_operators)} "
            f"unknowns={len(self.unknowns)}>"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def scan(sql, dialect=None):
    """Scan *sql* and return a :class:`ParameterInventory`.

    :param sql: The SQL text to scan.
    :param dialect: One of ``"postgresql"``, ``"mysql"``, ``"sqlite"``,
        ``"oracle"`` or ``None`` (unknown dialect).
    """
    return ParameterInventory(sql, dialect=dialect)
