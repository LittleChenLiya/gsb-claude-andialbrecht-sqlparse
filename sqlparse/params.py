#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Inventory of parameter placeholders in SQL scripts.

This module builds a static inventory of parameter placeholders
(named and positional) found in a SQL script.  It never executes SQL;
it only inspects the token stream produced by the sqlparse lexer.

Rules applied:

* Placeholders inside string literals, quoted identifiers, comments
  and PostgreSQL dollar-quoted literals are *not* parameters.  The
  sqlparse lexer already emits those regions as single non-placeholder
  tokens, so they are excluded by construction.
* The question mark ``?`` is ambiguous: in MySQL/SQLite/generic SQL it
  is a positional placeholder, in PostgreSQL it is the JSON existence
  operator.  Every ``?`` occurrence is listed separately in
  :attr:`ParamInventory.question_marks` with the dialect interpretation
  that was applied.
* Placeholder shapes that the configured dialect does not define are
  kept in the inventory with ``kind='unknown'`` instead of being
  silently dropped.
* Repeated references to the same named parameter are all preserved;
  use :meth:`ParamInventory.by_name` to group them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlparse
from sqlparse import tokens as T

#: Dialects with an explicit rule for the question mark token.
DIALECTS = ("generic", "mysql", "sqlite", "postgresql")

#: Placeholder shapes recognized on top of the lexer's Placeholder type.
_RE_NUMBERED = re.compile(r"\$\d+\Z")               # $1  (PostgreSQL)
_RE_PYFORMAT_NAMED = re.compile(r"%\((\w+)\)s\Z")   # %(name)s
_RE_NAMED = re.compile(r":\w+\Z")                   # :name

#: Dialects where ``$<digits>`` is a numbered positional placeholder.
_NUMBERED_DIALECTS = frozenset({"generic", "postgresql"})

#: Dialects where a bare ``?`` is a positional placeholder.  In
#: PostgreSQL ``?`` is the JSON existence operator instead.
_QMARK_PLACEHOLDER_DIALECTS = frozenset({"generic", "mysql", "sqlite"})

_QMARK_NOTES = {
    "generic": "positional placeholder (generic SQL)",
    "mysql": "positional placeholder (MySQL)",
    "sqlite": "positional placeholder (SQLite)",
    "postgresql": "JSON existence operator in PostgreSQL; "
                  "not a placeholder",
}


@dataclass(frozen=True)
class ParamRef:
    """One placeholder occurrence.

    :ivar statement: 1-based number of the statement containing it.
    :ivar order: 1-based occurrence order within the whole script.
    :ivar kind: ``'named'``, ``'positional'`` or ``'unknown'``.
    :ivar name: parameter name for named placeholders, else ``None``.
    :ivar text: original placeholder text as written in the script.
    :ivar span: ``(start, end)`` offsets into the original SQL text.
    :ivar note: dialect interpretation note, empty when unambiguous.
    """

    statement: int
    order: int
    kind: str
    name: str | None
    text: str
    span: tuple[int, int]
    note: str = ""


@dataclass
class ParamInventory:
    """Result of :func:`extract_params`.

    :ivar dialect: dialect the inventory was built with.
    :ivar params: all placeholder occurrences, in order of appearance.
    :ivar question_marks: every ``?`` occurrence with the dialect
        interpretation that was applied (also present in ``params``
        when the dialect treats it as a placeholder).
    """

    dialect: str
    params: list[ParamRef] = field(default_factory=list)
    question_marks: list[ParamRef] = field(default_factory=list)

    def by_name(self) -> dict[str, list[ParamRef]]:
        """Group named placeholders by name.

        Multiple references to the same name are all preserved.
        """
        grouped: dict[str, list[ParamRef]] = {}
        for ref in self.params:
            if ref.name is not None:
                grouped.setdefault(ref.name, []).append(ref)
        return grouped

    def render(self) -> str:
        """Render the inventory as a plain-text table."""
        lines = [
            f"dialect: {self.dialect}",
            "stmt  ord  kind        name      span          text",
        ]
        for ref in self.params:
            name = ref.name or "-"
            lines.append(
                f"{ref.statement:<5} {ref.order:<4} {ref.kind:<11} "
                f"{name:<9} {ref.span!s:<13} {ref.text}"
                + (f"   # {ref.note}" if ref.note else "")
            )
        if self.question_marks:
            lines.append("question marks (dialect interpretation):")
            for ref in self.question_marks:
                lines.append(
                    f"  stmt {ref.statement} span {ref.span!s}: {ref.note}"
                )
        return "\n".join(lines)


def _classify(text, dialect):
    """Return ``(kind, name, note)`` for one placeholder token."""
    if text == "?":
        note = _QMARK_NOTES[dialect]
        if dialect in _QMARK_PLACEHOLDER_DIALECTS:
            return "positional", None, note
        return "unknown", None, note
    if _RE_NAMED.match(text):
        return "named", text[1:], ""
    if _RE_NUMBERED.match(text):
        if dialect in _NUMBERED_DIALECTS:
            return "positional", None, f"numbered placeholder {text}"
        return "unknown", None, f"{text} is not a placeholder in {dialect}"
    m = _RE_PYFORMAT_NAMED.match(text)
    if m:
        return "named", m.group(1), "Python pyformat placeholder"
    if text == "%s":
        return "positional", None, "Python format placeholder"
    return "unknown", None, f"unrecognized placeholder shape in {dialect}"


def extract_params(sql_text, dialect="generic"):
    """Build a placeholder inventory for *sql_text*.

    :param sql_text: one or more SQL statements; never executed.
    :param dialect: one of ``'generic'``, ``'mysql'``, ``'sqlite'`` or
        ``'postgresql'``; controls the interpretation of ``?`` and
        ``$<digits>``.
    :returns: a :class:`ParamInventory`.
    """
    if dialect not in DIALECTS:
        raise ValueError(
            f"unknown dialect {dialect!r}; expected one of {DIALECTS}")

    inventory = ParamInventory(dialect=dialect)
    order = 0
    search_from = 0
    for stmt_no, statement in enumerate(sqlparse.parse(sql_text), 1):
        stmt_text = str(statement)
        # Locate the statement in the original text.  Statements are
        # emitted in order and verbatim, so a forward search is exact.
        stmt_start = sql_text.find(stmt_text, search_from)
        if stmt_start < 0:  # pragma: no cover - defensive
            stmt_start = search_from
        search_from = stmt_start + len(stmt_text)

        offset = 0
        for token in statement.flatten():
            start = stmt_start + offset
            offset += len(token.value)
            if token.ttype is not T.Name.Placeholder:
                continue
            order += 1
            kind, name, note = _classify(token.value, dialect)
            ref = ParamRef(
                statement=stmt_no,
                order=order,
                kind=kind,
                name=name,
                text=token.value,
                span=(start, start + len(token.value)),
                note=note,
            )
            inventory.params.append(ref)
            if token.value == "?":
                inventory.question_marks.append(ref)
    return inventory
