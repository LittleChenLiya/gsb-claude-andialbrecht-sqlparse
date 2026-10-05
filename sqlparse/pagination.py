#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Explicit pagination rewrite for the outermost SELECT.

This module adds or replaces the LIMIT/OFFSET clause of the
outermost SELECT of a single SQL statement.  Subqueries, CTEs,
string literals and comments are never touched: the rewrite is
performed as a byte-level splice of the original statement text, so
every unmodified region keeps its exact original bytes.

Before any modification the complete statement is validated:

* multi-statement input is rejected (the first statement is not
  silently modified),
* statements that are not a SELECT are rejected,
* UNION queries are rejected,
* statements using an unsupported pagination dialect (TOP,
  FETCH FIRST, ROWNUM) are rejected.

The existing outermost pagination is first parsed into a
PaginationPlan that records the exact values and their
placeholder identity.

Examples
--------

Add a LIMIT to a statement without pagination::

    >>> rewrite_pagination('SELECT * FROM t', limit=10)
    'SELECT * FROM t LIMIT 10'

Replace an existing LIMIT, keeping the inner query bytes intact::

    >>> rewrite_pagination(
    ...     'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 10',
    ...     limit=20)
    'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 20'

Comments and string literals containing "LIMIT" do not participate::

    >>> rewrite_pagination('SELECT * FROM t -- LIMIT 999', limit=10)
    'SELECT * FROM t LIMIT 10 -- LIMIT 999'

Parameterized pagination keeps its placeholder identity::

    >>> rewrite_pagination('SELECT * FROM t LIMIT ?', offset=5)
    'SELECT * FROM t LIMIT ? OFFSET 5'

UNION queries and multi-statement input are rejected::

    >>> rewrite_pagination('SELECT * FROM a UNION SELECT * FROM b', limit=10)
    Traceback (most recent call last):
        ...
    UnsupportedStatementError: UNION queries are not supported

    >>> rewrite_pagination('SELECT * FROM a; SELECT * FROM b', limit=10)
    Traceback (most recent call last):
        ...
    UnsupportedStatementError: expected a single SQL statement, got 2

Run the tests with::

    python -m pytest tests/test_pagination.py -v
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import imt

__all__ = [
    'PaginationError',
    'PaginationPlan',
    'PaginationValue',
    'UnsupportedDialectError',
    'UnsupportedStatementError',
    'parse_pagination',
    'rewrite_pagination',
]


class PaginationError(ValueError):
    """Base class for pagination rewrite errors."""


class UnsupportedStatementError(PaginationError):
    """The input is not a single, supported outermost SELECT."""


class UnsupportedDialectError(PaginationError):
    """The statement uses a pagination dialect that is not supported."""


@dataclass(frozen=True)
class PaginationValue:
    """A parsed LIMIT or OFFSET value."""
    value: str
    is_placeholder: bool


@dataclass(frozen=True)
class PaginationPlan:
    """Parsed pagination of the outermost SELECT."""
    limit: PaginationValue | None
    offset: PaginationValue | None
    span: tuple[int, int] | None
    dialect: str = 'limit-offset'


# Placeholder shapes accepted by the lexer: '?', '%s', '%(name)s',
# and '$name'/':name'/'?name'.
_PLACEHOLDER_RE = re.compile(r'^(?:\?|%\(\w+\)s|%s|[$:?]\w+)$')


def _coerce_value(value, name):
    """Coerce a user supplied limit/offset value to a PaginationValue."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an int, str or None, got bool')
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f'{name} must be non-negative, got {value}')
        return PaginationValue(str(value), False)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f'{name} must not be empty')
        if _PLACEHOLDER_RE.match(text):
            return PaginationValue(text, True)
        if text.isdigit():
            return PaginationValue(text, False)
        raise ValueError(
            f'{name} must be an integer literal or a parameter placeholder '
            f'(e.g. "?", ":name", "%s"), got {value!r}')
    raise TypeError(
        f'{name} must be an int, str or None, got {type(value).__name__}')


def _is_comment(token):
    """Return True if token is a comment (group or plain token)."""
    return imt(token, i=sql.Comment) or token.ttype in T.Comment


def _validate_statement(stmt):
    """Validate that stmt is a single outermost SELECT without UNION."""
    if stmt.get_type() != 'SELECT':
        raise UnsupportedStatementError(
            f'only SELECT statements are supported, '
            f'got {stmt.get_type()!r}')
    # UNION / UNION ALL at the outermost level
    for token in stmt.tokens:
        if token.is_keyword and token.normalized.startswith('UNION'):
            raise UnsupportedStatementError(
                'UNION queries are not supported')


def _detect_top(stmt):
    """Detect MSSQL/Sybase TOP n pagination after SELECT."""
    select_idx = None
    for idx, token in enumerate(stmt.tokens):
        if token.is_keyword and token.normalized == 'SELECT':
            select_idx = idx
            break
    if select_idx is None:
        return
    first_idx, first = stmt.token_next(select_idx, skip_ws=True)
    if first is None:
        return
    if isinstance(first, sql.Identifier) and first.value.upper() == 'TOP':
        _, nxt = stmt.token_next(first_idx, skip_ws=True)
        if nxt is not None and (
                nxt.ttype in T.Literal.Number
                or nxt.ttype is T.Name.Placeholder):
            raise UnsupportedDialectError(
                'TOP pagination is not supported; '
                'use LIMIT/OFFSET instead')


def _has_rownum(tokens):
    """Return True if ROWNUM appears outside of parentheses."""
    for token in tokens:
        if isinstance(token, sql.Parenthesis):
            continue
        if token.ttype is T.Name.Builtin and token.value.upper() == 'ROWNUM':
            return True
        if token.is_group:
            if _has_rownum(token.tokens):
                return True
    return False


def _detect_unsupported_dialect(stmt):
    """Reject pagination dialects that are not LIMIT/OFFSET."""
    # FETCH FIRST/NEXT n ROWS ONLY (SQL standard, DB2, modern MSSQL/Oracle)
    for token in stmt.tokens:
        if token.is_keyword and token.normalized == 'FETCH':
            raise UnsupportedDialectError(
                'FETCH FIRST/NEXT pagination is not supported; '
                'use LIMIT/OFFSET instead')
    _detect_top(stmt)
    if _has_rownum(stmt.tokens):
        raise UnsupportedDialectError(
            'ROWNUM pagination is not supported; use LIMIT/OFFSET instead')


def _find_pagination_clauses(stmt):
    """Return the indices of the outermost LIMIT and OFFSET keywords."""
    limit_idx = None
    offset_idx = None
    for idx, token in enumerate(stmt.tokens):
        if token.is_keyword:
            norm = token.normalized
            if norm == 'LIMIT':
                limit_idx = idx
            elif norm == 'OFFSET':
                offset_idx = idx
    return limit_idx, offset_idx


def _parse_single_value(token):
    """Parse a single LIMIT or OFFSET value token."""
    if token.ttype in T.Literal.Number:
        return PaginationValue(token.value, False)
    if token.ttype is T.Name.Placeholder:
        return PaginationValue(token.value, True)
    if token.is_keyword:
        # e.g. LIMIT ALL (PostgreSQL)
        return PaginationValue(token.value, False)
    raise UnsupportedDialectError(
        f'unsupported pagination value: {token.value!r}')


def _parse_limit_value(token):
    """Parse a LIMIT value, handling MySQL offset, count style.

    Returns a tuple (limit, offset) where either may be None.
    """
    if isinstance(token, sql.IdentifierList):
        parts = [t for t in token.tokens
                 if not t.is_whitespace
                 and not t.match(T.Punctuation, ',')]
        if len(parts) == 2:
            offset_val = _parse_single_value(parts[0])
            count_val = _parse_single_value(parts[1])
            return count_val, offset_val
    return _parse_single_value(token), None


def _value_index_after(children, kw_idx):
    """Return the index of the first non-whitespace child after kw_idx."""
    for j in range(kw_idx + 1, len(children)):
        token = children[j]
        if not token.is_whitespace and not _is_comment(token):
            return j
    return None


def _compute_span(stmt, limit_idx, offset_idx):
    """Compute the byte span of the outermost pagination clause."""
    if limit_idx is None and offset_idx is None:
        return None

    children = []
    offset = 0
    for token in stmt.tokens:
        start = offset
        offset += len(token.value)
        children.append((token, start, offset))

    first_kw_idx = limit_idx if limit_idx is not None else offset_idx
    last_value_idx = None
    if limit_idx is not None:
        v = _value_index_after([c[0] for c in children], limit_idx)
        if v is not None:
            last_value_idx = v
    if offset_idx is not None:
        v = _value_index_after([c[0] for c in children], offset_idx)
        if v is not None:
            last_value_idx = (
                v if last_value_idx is None else max(last_value_idx, v))

    if last_value_idx is None:
        return None

    return (children[first_kw_idx][1], children[last_value_idx][2])


def _parse_plan(stmt):
    """Parse the outermost pagination of a validated SELECT statement."""
    limit_idx, offset_idx = _find_pagination_clauses(stmt)

    limit_value = None
    offset_value = None

    if limit_idx is not None:
        children = list(stmt.tokens)
        v_idx = _value_index_after(children, limit_idx)
        if v_idx is not None:
            limit_value, comma_offset = _parse_limit_value(children[v_idx])
            if comma_offset is not None and offset_idx is None:
                offset_value = comma_offset

    if offset_idx is not None:
        children = list(stmt.tokens)
        v_idx = _value_index_after(children, offset_idx)
        if v_idx is not None:
            offset_value = _parse_single_value(children[v_idx])

    span = _compute_span(stmt, limit_idx, offset_idx)
    return PaginationPlan(
        limit=limit_value, offset=offset_value, span=span)


def _find_insertion_point(stmt):
    """Find the byte offset at which to append a new pagination clause.

    The clause is inserted after the last token of the outermost
    SELECT body, before trailing whitespace/comments/semicolons and
    before locking clauses such as FOR UPDATE.
    """
    insert_at = 0
    offset = 0
    for token in stmt.tokens:
        end = offset + len(token.value)
        if token.is_keyword and token.normalized in ('FOR', 'LOCK'):
            return insert_at
        if (not token.is_whitespace
                and not _is_comment(token)
                and not token.match(T.Punctuation, ';')):
            insert_at = end
        offset = end
    return insert_at


def parse_pagination(sql_text):
    """Parse the outermost SELECT's existing pagination into a plan.

    The complete statement is validated first: multi-statement input,
    non-SELECT statements, UNION queries and unsupported pagination
    dialects are rejected with a PaginationError.

    :param sql_text: a string containing a single SQL statement
    :returns: a PaginationPlan
    """
    stmts = sqlparse.parse(sql_text)
    if len(stmts) != 1:
        raise UnsupportedStatementError(
            f'expected a single SQL statement, got {len(stmts)}')
    stmt = stmts[0]
    _validate_statement(stmt)
    _detect_unsupported_dialect(stmt)
    return _parse_plan(stmt)


def rewrite_pagination(sql_text, limit=None, offset=None):
    """Add or replace the outermost SELECT's LIMIT/OFFSET.

    The complete statement is validated before any modification.
    Multi-statement input, non-SELECT statements, UNION queries and
    unsupported pagination dialects are rejected with a
    PaginationError.

    Unmodified regions keep their exact original bytes; only the
    outermost pagination clause is added or replaced.

    :param sql_text: a string containing a single SQL statement
    :param limit: the new LIMIT value (an int literal, a placeholder
        string such as '?', ':name' or '%s', or None to keep the
        existing value)
    :param offset: the new OFFSET value (same rules as limit)
    :returns: the rewritten SQL statement as a string
    """
    plan = parse_pagination(sql_text)

    new_limit = _coerce_value(limit, 'limit')
    new_offset = _coerce_value(offset, 'offset')

    final_limit = new_limit if new_limit is not None else plan.limit
    final_offset = new_offset if new_offset is not None else plan.offset

    parts = []
    if final_limit is not None:
        parts.append('LIMIT ' + final_limit.value)
    if final_offset is not None:
        parts.append('OFFSET ' + final_offset.value)
    new_clause = ' '.join(parts)

    if plan.span is not None:
        start, end = plan.span
        return sql_text[:start] + new_clause + sql_text[end:]

    if not new_clause:
        return sql_text

    insert_at = _find_insertion_point(sqlparse.parse(sql_text)[0])
    return sql_text[:insert_at] + ' ' + new_clause + sql_text[insert_at:]
