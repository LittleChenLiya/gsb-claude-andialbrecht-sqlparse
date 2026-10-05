"""Explicit pagination rewriting for the outermost ``SELECT`` statement.

This module adds or replaces only the ``LIMIT`` / ``OFFSET`` clause of the
outermost query.  It never rewrites subqueries, CTE bodies, comments or
string literals.  Rewriting is planned in two phases:

1. :func:`plan_pagination` parses the *complete* input statement(s) into an
   explicit :class:`PaginationPlan` whose edits are byte offsets into the
   original text.  Existing pagination is parsed first (including parameter
   placeholders, which keep their identity), and every rejection is raised
   here -- before any byte is touched.
2. :func:`PaginationPlan.apply` (or the convenience helper
   :func:`paginate`) applies those edits with plain string slicing, so every
   unmodified region keeps its original bytes.  No full-text regular
   expressions are used.

Supported dialects: ``postgresql``, ``mysql``, ``sqlite``, ``mariadb``,
``snowflake`` (all render ``LIMIT n OFFSET m``).  Unsupported
dialects, set operations (``UNION``/``INTERSECT``/``EXCEPT``/``MINUS``),
non-SELECT statements, multi-statement input and the non-LIMIT pagination
forms ``TOP`` / ``FETCH FIRST`` are rejected explicitly instead of being
guessed at.

Before / after
--------------
>>> from sqlparse.pagination import plan_pagination
>>> # add LIMIT 10 to the outer query; the subquery LIMIT 5 stays byte-identical
>>> plan_pagination(
...     "SELECT * FROM (SELECT a FROM t LIMIT 5) x", 10
... ).apply()
'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 10'
>>> # existing pagination is parsed first, then replaced
>>> plan_pagination("SELECT * FROM t LIMIT 20", 30, 5).apply()
'SELECT * FROM t LIMIT 30 OFFSET 5'
>>> # parameter placeholders keep their identity unless overridden
>>> plan_pagination("SELECT * FROM t LIMIT %s OFFSET %s").apply()
'SELECT * FROM t LIMIT %s OFFSET %s'
>>> # set operations and multi-statement input are rejected, not guessed
>>> plan_pagination("SELECT * FROM a UNION SELECT * FROM b", 10)
Traceback (most recent call last):
    ...
sqlparse.pagination.UnsupportedPaginationError: set operation UNION ...

Tests
-----
Run the focused suite with::

    python -m pytest tests/test_pagination.py -q

and the runnable before/after examples with::

    PYTHONPATH=. python examples/pagination_rewrite.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlparse as _sqlparse
from sqlparse import sql
from sqlparse import tokens as T

# Dialects that render pagination as ``LIMIT n [OFFSET m]``.
SUPPORTED_DIALECTS: frozenset[str] = frozenset(
    {
        "postgresql",
        "postgres",
        "mysql",
        "sqlite",
        "mariadb",
        "snowflake",
    }
)

# Set operators that make "outermost SELECT" ambiguous -- never rewritten.
_SET_OPERATORS: frozenset[str] = frozenset(
    {"UNION", "UNION ALL", "INTERSECT", "EXCEPT", "MINUS"}
)

# Parameter placeholders as produced by the lexer or as a plain ``@p1`` name.
_PLACEHOLDER_RE = re.compile(
    r"%(\(\w+\))?s$"  # psycopg/DB-API: %s, %(name)s
    r"|^\$\d+$"  # postgres: $1
    r"|^\?"  # sqlite/qmark: ?, ?1
    r"|^[:@][A-Za-z_][A-Za-z0-9_]*$"  # named: :name, @name
)


class PaginationError(ValueError):
    """Base class for pagination rewriting errors."""


class UnsupportedPaginationError(PaginationError):
    """Raised when a statement cannot be paginated explicitly.

    This covers set operations (``UNION`` ...), statements that are not an
    outermost ``SELECT``, unsupported dialects and pagination constructs
    that are not ``LIMIT``/``OFFSET`` based (``TOP``, ``FETCH FIRST``).
    """


class MultipleStatementsError(PaginationError):
    """Raised when more than one statement would have to be considered.

    The first statement is *never* rewritten silently; the caller has to
    make the statement boundary explicit.
    """


@dataclass(frozen=True)
class Placeholder:
    """A parameter marker used in an existing or requested pagination clause.

    Placeholders are compared by their textual identity (``%s``, ``$1``,
    ``:limit`` ...) so a rewrite can preserve them instead of substituting
    a value.
    """

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class LimitValue:
    """One parsed value position inside a pagination clause."""

    placeholder: Placeholder | None
    integer: int | None
    # Byte span of the value inside the full input string.
    start: int
    end: int

    @property
    def is_placeholder(self) -> bool:
        return self.placeholder is not None

    def render(self, replacement: int | Placeholder | None = None) -> str:
        if replacement is None:
            if self.placeholder is not None:
                return self.placeholder.text
            return str(self.integer)
        return str(replacement)


@dataclass(frozen=True)
class _Edit:
    """A single byte-range edit, expressed against the original text."""

    start: int
    end: int
    text: str  # replacement text; ``end == start`` means a pure insertion


@dataclass(frozen=True)
class PaginationPlan:
    """The explicit, pre-computed result of pagination planning.

    Attributes:
        original_sql: The full input text, kept for byte-accurate applying.
        dialect: Dialect name the plan was built for.
        limit: Parsed outer limit value (``None`` when the query had none).
        offset: Parsed outer offset value (``None`` when the query had none).
        comma_style: True for MySQL's ``LIMIT off, lim`` spelling.
        edits: Ordered, non-overlapping byte edits against *original_sql*.
        statement_span: ``(start, end)`` byte span of the single statement.
    """

    original_sql: str
    dialect: str
    limit: LimitValue | None
    offset: LimitValue | None
    comma_style: bool
    edits: tuple[_Edit, ...] = field(default=())
    statement_span: tuple[int, int] = (0, 0)

    @property
    def has_pagination(self) -> bool:
        """True when the outermost statement already carried pagination."""
        return self.limit is not None or self.offset is not None

    def apply(self) -> str:
        """Apply edits to the original text and return the new SQL.

        Edits are applied strictly backwards (highest offset first), so plain
        slicing never invalidates earlier offsets and every byte outside an
        edit range is preserved verbatim.
        """
        result = self.original_sql
        for edit in sorted(self.edits, key=lambda e: (e.start, e.end), reverse=True):
            result = result[: edit.start] + edit.text + result[edit.end :]
        return result

    def as_text(self) -> str:
        """Human-readable description, useful for logging/auditing."""
        parts = []
        for edit in self.edits:
            kind = "insert" if edit.start == edit.end else "replace"
            parts.append(f"{kind}@{edit.start}:{edit.end} -> {edit.text!r}")
        return "; ".join(parts) or "no-op"


# ---------------------------------------------------------------------------
# Token scanning helpers
# ---------------------------------------------------------------------------


def _statement_bounds(
    sql_text: str, statements: tuple[sql.Statement, ...]
) -> tuple[int, int, sql.Statement]:
    """Return ``(start, end, statement)`` for the only real statement.

    sqlparse statements partition the input exactly (the sum of their string
    values equals the input).  Statements consisting solely of whitespace or
    comments are ignored; two or more real statements are rejected so that a
    multi-statement string is never handled by silently editing the first.
    """
    base = 0
    real: list[tuple[int, int, sql.Statement]] = []
    for stmt in statements:
        text = str(stmt)
        end = base + len(text)
        has_real_token = any(
            not tok.is_whitespace and not _is_comment(tok.ttype)
            for tok in stmt.flatten()
        )
        if has_real_token:
            real.append((base, end, stmt))
        base = end

    if not real:
        raise PaginationError("empty statement: nothing to paginate")
    if len(real) > 1:
        raise MultipleStatementsError(
            f"expected exactly one statement, found {len(real)}; "
            "the first statement is not rewritten silently"
        )
    return real[0]


def _top_level_tokens(
    stmt: sql.Statement, base: int
) -> list[tuple[sql.Token, int, int]]:
    """Return ``(token, start, end)`` for the statement's top-level tokens.

    End offsets are derived by summing token values (sqlparse never drops or
    changes a byte), so no text search is involved.
    """
    result: list[tuple[sql.Token, int, int]] = []
    pos = base
    for tok in stmt.tokens:
        result.append((tok, pos, pos + len(tok.value)))
        pos += len(tok.value)
    return result


def _leaf_spans(tok: sql.Token, start: int) -> list[tuple[sql.Token, int, int]]:
    """Flatten *tok* and return leaf tokens with their absolute spans."""
    out: list[tuple[sql.Token, int, int]] = []
    pos = start
    for leaf in tok.flatten():
        out.append((leaf, pos, pos + len(leaf.value)))
        pos += len(leaf.value)
    return out


def _is_comment(ttype) -> bool:
    return ttype in (T.Comment, T.Comment.Single, T.Comment.Multiline)


def _significant(tok: sql.Token) -> bool:
    return not tok.is_whitespace and not _is_comment(tok.ttype)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def _as_placeholder(tok: sql.Token) -> Placeholder | None:
    if tok.ttype == T.Name.Placeholder:
        return Placeholder(tok.value)
    # ``@p1`` is lexed as a plain Name.
    if tok.ttype == T.Name and _PLACEHOLDER_RE.match(tok.value):
        return Placeholder(tok.value)
    return None


def _value_from_token(
    tok: sql.Token, start: int, end: int, what: str
) -> LimitValue:
    """Parse a single-token (possibly grouped) limit/offset value."""
    leaves = [
        (leaf, ls, le)
        for leaf, ls, le in _leaf_spans(tok, start)
        if _significant(leaf)
    ]
    if len(leaves) == 1:
        leaf, ls, le = leaves[0]
        ph = _as_placeholder(leaf)
        if ph is not None:
            return LimitValue(ph, None, ls, le)
        if leaf.ttype == T.Literal.Number.Integer:
            return LimitValue(None, int(leaf.value), ls, le)
    raise UnsupportedPaginationError(
        f"unsupported {what} value: {tok.value!r} (only integer literals "
        "and parameter placeholders are supported)"
    )


def _values_from_comma_group(
    tok: sql.Token, start: int
) -> tuple[LimitValue, LimitValue]:
    """Parse a grouped MySQL ``LIMIT off, lim`` value list."""
    leaves = [
        (leaf, ls, le)
        for leaf, ls, le in _leaf_spans(tok, start)
        if _significant(leaf)
    ]
    if (
        len(leaves) != 3
        or leaves[1][0].ttype != T.Punctuation
        or leaves[1][0].value != ","
    ):
        raise UnsupportedPaginationError(
            f"unsupported comma-style LIMIT value: {tok.value!r}"
        )

    def one(item):
        leaf, ls, le = item
        ph = _as_placeholder(leaf)
        if ph is not None:
            return LimitValue(ph, None, ls, le)
        if leaf.ttype == T.Literal.Number.Integer:
            return LimitValue(None, int(leaf.value), ls, le)
        raise UnsupportedPaginationError(
            f"unsupported comma-style LIMIT value: {tok.value!r}"
        )

    return one(leaves[0]), one(leaves[2])


# ---------------------------------------------------------------------------
# Outer-clause scanning
# ---------------------------------------------------------------------------


@dataclass
class _Scan:
    limit_kw: tuple[int, int] | None = None
    offset_kw: tuple[int, int] | None = None
    comma_style: bool = False
    limit: LimitValue | None = None
    offset: LimitValue | None = None
    insert_anchor: int = -1  # insert point for a new clause
    before_semicolon: bool = False  # anchor is just before a trailing ";"


def _scan_statement(
    stmt: sql.Statement, base: int, span_end: int
) -> _Scan:
    tops = _top_level_tokens(stmt, base)
    sig = [(t, s, e) for t, s, e in tops if _significant(t)]
    scan = _Scan()

    # --- reject anything that is not a plain outermost SELECT -------------
    first_tok = sig[0][0]
    if first_tok.ttype == T.Keyword.CTE:
        dml = next(
            (t for t, _, _ in sig if t.ttype == T.Keyword.DML), None
        )
        if dml is None or dml.normalized.upper() != "SELECT":
            raise UnsupportedPaginationError(
                "only an outermost SELECT (optionally preceded by a WITH "
                "clause) can be paginated"
            )
    elif (
        first_tok.ttype != T.Keyword.DML
        or first_tok.normalized.upper() != "SELECT"
    ):
        raise UnsupportedPaginationError(
            "only an outermost SELECT can be paginated, got: "
            f"{first_tok.normalized!r}"
        )

    # --- reject set operations and non-LIMIT pagination forms -------------
    for tok, _, _ in sig:
        if (
            tok.ttype in (T.Keyword, T.Keyword.DML)
            and tok.normalized.upper() in _SET_OPERATORS
        ):
            raise UnsupportedPaginationError(
                f"set operation {tok.normalized.upper()} is not supported; "
                "wrap the query or paginate a single SELECT"
            )
        if tok.ttype == T.Keyword and tok.normalized.upper() == "FETCH":
            raise UnsupportedPaginationError(
                "FETCH FIRST / NEXT pagination is not supported"
            )
        # SELECT TOP n ... -- TOP is lexed as a plain Name and grouped.
        if tok.is_group:
            inner = next(iter(tok.flatten()), None)
            if (
                inner is not None
                and inner.ttype == T.Name
                and inner.normalized.upper() == "TOP"
            ):
                raise UnsupportedPaginationError(
                    "TOP pagination is not supported"
                )

    # --- locate a top-level LIMIT keyword with a following value ----------
    for i, (tok, s, e) in enumerate(sig):
        if tok.ttype == T.Keyword and tok.normalized.upper() == "LIMIT":
            if i + 1 >= len(sig):
                continue  # keyword without a value: not a pagination clause
            value, vs, ve = sig[i + 1]
            is_value = (
                value.ttype == T.Literal.Number.Integer
                or value.ttype == T.Name.Placeholder
                or value.is_group
            )
            if not is_value:
                # ``SELECT limit FROM ...`` etc.
                continue
            scan.limit_kw = (s, e)
            if value.is_group and any(
                leaf.ttype == T.Punctuation and leaf.value == ","
                for leaf in value.flatten()
            ):
                off_val, lim_val = _values_from_comma_group(value, vs)
                scan.comma_style = True
                scan.offset = off_val
                scan.limit = lim_val
            else:
                scan.limit = _value_from_token(value, vs, ve, "LIMIT")
            break

    # --- locate a top-level OFFSET keyword with a following value ---------
    for i, (tok, s, e) in enumerate(sig):
        if tok.ttype == T.Keyword and tok.normalized.upper() == "OFFSET":
            if i + 1 >= len(sig):
                raise UnsupportedPaginationError("OFFSET without a value")
            vt, vs, ve = sig[i + 1]
            scan.offset_kw = (s, e)
            scan.offset = _value_from_token(vt, vs, ve, "OFFSET")
            break

    # --- insertion point for a brand new clause ---------------------------
    for tok, s, _e in sig:
        if tok.ttype == T.Keyword and tok.normalized.upper() == "FOR":
            scan.insert_anchor = s
            return scan
    for tok, s, _e in reversed(tops):
        if tok.ttype == T.Punctuation and tok.value == ";":
            scan.insert_anchor = s
            scan.before_semicolon = True
            return scan
    scan.insert_anchor = span_end
    return scan


# ---------------------------------------------------------------------------
# Request handling & public API
# ---------------------------------------------------------------------------


def _coerce_request(value, name: str) -> int | Placeholder:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or Placeholder, got bool")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value
    if isinstance(value, Placeholder):
        return value
    if isinstance(value, str) and _PLACEHOLDER_RE.match(value):
        return Placeholder(value)
    raise TypeError(
        f"{name} must be an int or Placeholder, got {type(value).__name__}"
    )


def _render_clause(limit_text: str, offset_text: str | None) -> str:
    clause = f"LIMIT {limit_text}"
    if offset_text is not None:
        clause += f" OFFSET {offset_text}"
    return clause


def plan_pagination(
    sql_text: str,
    limit: int | Placeholder | None = None,
    offset: int | Placeholder | None = None,
    *,
    dialect: str = "postgresql",
) -> PaginationPlan:
    """Parse *sql_text* and build an explicit pagination rewrite plan.

    The complete input is parsed and validated before the plan is returned;
    this function never performs a rewrite itself.  Existing LIMIT/OFFSET
    values are parsed into the plan (parameter placeholders keep their
    identity); arguments left as ``None`` mean "keep the existing value".

    :raises UnsupportedPaginationError: UNION/INTERSECT/EXCEPT/MINUS queries,
        non-SELECT statements, TOP/FETCH FIRST pagination, unsupported
        dialects or unparseable values.
    :raises MultipleStatementsError: more than one statement was given.
    """
    if not isinstance(sql_text, str):
        raise TypeError("sql_text must be a string")
    if dialect not in SUPPORTED_DIALECTS:
        raise UnsupportedPaginationError(
            f"dialect {dialect!r} is not supported; supported dialects: "
            f"{', '.join(sorted(SUPPORTED_DIALECTS))}"
        )

    req_limit = _coerce_request(limit, "limit") if limit is not None else None
    req_offset = (
        _coerce_request(offset, "offset") if offset is not None else None
    )

    statements = tuple(_sqlparse.parse(sql_text))
    joined = "".join(str(stmt) for stmt in statements)
    if joined != sql_text:  # pragma: no cover - sqlparse preserves bytes
        raise PaginationError("parser did not preserve input bytes")

    start, end, stmt = _statement_bounds(sql_text, statements)
    scan = _scan_statement(stmt, start, end)

    if scan.limit is None and req_limit is None:
        raise PaginationError(
            "statement has no LIMIT clause; pass an explicit limit to add one"
        )

    edits: list[_Edit] = []

    if scan.limit is not None:
        # --- existing pagination: replace value tokens in place ----------
        new_limit = scan.limit.render(req_limit)
        old_limit = sql_text[scan.limit.start : scan.limit.end]
        if new_limit != old_limit:
            edits.append(_Edit(scan.limit.start, scan.limit.end, new_limit))

        if scan.offset is not None:
            new_off = scan.offset.render(req_offset)
            old_off = sql_text[scan.offset.start : scan.offset.end]
            if new_off != old_off:
                edits.append(
                    _Edit(scan.offset.start, scan.offset.end, new_off)
                )
        elif not scan.comma_style and req_offset is not None:
            # Existing ``LIMIT n`` gains a trailing `` OFFSET m``.
            insert_at = scan.limit.end
            edits.append(
                _Edit(insert_at, insert_at, f" OFFSET {req_offset}")
            )
    else:
        # --- no pagination yet: insert one complete clause ----------------
        clause = _render_clause(
            str(req_limit),
            str(req_offset) if req_offset is not None else None,
        )
        anchor = scan.insert_anchor
        prefix = ""
        if anchor > start and not sql_text[start:anchor][-1].isspace():
            prefix = " "
        suffix = ""
        # When inserting immediately before a following keyword (e.g.
        # ``FOR UPDATE``), separate the new clause from it as well.  A
        # trailing semicolon needs no separator.
        if (
            not scan.before_semicolon
            and anchor < end
            and not sql_text[anchor].isspace()
        ):
            suffix = " "
        edits.append(_Edit(anchor, anchor, prefix + clause + suffix))

    edits.sort(key=lambda e: (e.start, e.end))
    return PaginationPlan(
        original_sql=sql_text,
        dialect=dialect,
        limit=scan.limit,
        offset=scan.offset,
        comma_style=scan.comma_style,
        edits=tuple(edits),
        statement_span=(start, end),
    )


def paginate(
    sql_text: str,
    limit: int | Placeholder | None = None,
    offset: int | Placeholder | None = None,
    *,
    dialect: str = "postgresql",
) -> str:
    """Convenience helper: build the plan and apply it in one call."""
    return plan_pagination(
        sql_text, limit, offset, dialect=dialect
    ).apply()
