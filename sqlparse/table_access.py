"""Offline table-access report built on top of sqlparse (no database needed).

This module inspects SQL text with sqlparse's token tree and reports, for
every statement, which tables are actually read and which are written,
together with the original text span and the subquery scope each access
belongs to.

Supported grammar (explicitly, nothing more is claimed):

* ``SELECT`` (including subqueries in ``FROM``/``WHERE``/select-list and
  set operations such as ``UNION``)
* ``INSERT ... INTO ... [SELECT|VALUES]``
* ``UPDATE ... [FROM ...]``
* ``DELETE FROM ... [USING ...]``
* ``JOIN`` of all flavours and comma-joins
* ``WITH`` common table expressions, including ``WITH RECURSIVE``

Semantics implemented here:

* A CTE name is a *logical* source, visible only in the scope where the
  ``WITH`` clause applies (its own body for recursive self-reference, later
  CTE bodies and the main query).  References to it are reported with
  ``source='cte'`` and never as physical tables -- in particular a
  recursive CTE's self-reference is not reported as a physical table.
* Column qualifiers (``x.id``) and aliases (``FROM t x``) are never
  reported as table names: only the leading ``name[.name...]`` chain of a
  table reference after ``FROM``/``JOIN``/``INTO``/``UPDATE`` is used.
* Schema-qualified and quoted identifiers keep their real segments:
  ``"My Schema"."Order Details"`` is reported as the two segments
  ``My Schema`` / ``Order Details`` with quotes stripped, while ``raw``
  keeps the exact source text.
* Anything outside the supported grammar (``MERGE``, ``LATERAL``,
  table-valued functions, ``PIVOT``, ``ON CONFLICT`` ...) produces an
  explicit ``unknowns`` entry.  The report never claims a complete parse.

Attribution line format (see :func:`format_report`)::

    <scope> | <access> | <source> | <table> | span=<start>:<end> | raw=<raw text>

where ``access`` is ``read`` or ``write`` and ``source`` is ``physical``
or ``cte``.  ``scope`` labels look like ``S1`` (main query of statement 1),
``S1.q2`` (second subquery scope) or ``S1.cte:t`` (body of CTE ``t``).

Run ``python -m sqlparse.table_access`` for three short demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import (
    Function,
    Identifier,
    IdentifierList,
    Parenthesis,
    TokenList,
)

__all__ = [
    'Report',
    'StatementReport',
    'TableAccess',
    'UnknownConstruct',
    'analyze',
    'format_report',
]

#: Statement types this module fully understands.
SUPPORTED_TYPES = frozenset({'SELECT', 'INSERT', 'UPDATE', 'DELETE'})

#: Token types that may appear as one segment of a (possibly quoted) name.
_NAME_TTYPES = (T.Name, T.Name.Quoted, T.String.Symbol, T.Keyword)

#: Keyword fragments that mark dialect constructs we do not model.
_UNKNOWN_KEYWORDS = ('LATERAL', 'PIVOT', 'UNPIVOT', 'ON CONFLICT',
                     'ON DUPLICATE')


def _unquote(segment):
    """Strip one layer of identifier quotes, keeping the real segment."""
    if len(segment) >= 2:
        if segment[0] == segment[-1] and segment[0] in '"`':
            return segment[1:-1].replace(segment[0] * 2, segment[0])
        if segment[0] == '[' and segment[-1] == ']':
            return segment[1:-1]
    return segment


@dataclass
class TableAccess:
    """One read or write of a table inside a statement."""

    table: str        # normalized dotted name, quotes stripped
    raw: str          # exact source text of the name (quotes preserved)
    access: str       # 'read' | 'write'
    source: str       # 'physical' | 'cte'
    scope: str        # e.g. 'S1', 'S1.q2', 'S1.cte:t'
    span: tuple       # (start, end) offsets into the original SQL text


@dataclass
class UnknownConstruct:
    """A construct outside the supported grammar; parsing is incomplete."""

    reason: str
    scope: str
    span: tuple
    text: str


@dataclass
class StatementReport:
    """Access report for a single statement."""

    label: str                       # e.g. 'S1'
    type: str                        # SELECT/INSERT/UPDATE/DELETE/UNKNOWN...
    span: tuple
    accesses: list = field(default_factory=list)
    unknowns: list = field(default_factory=list)

    @property
    def reads(self):
        return [a for a in self.accesses if a.access == 'read']

    @property
    def writes(self):
        return [a for a in self.accesses if a.access == 'write']

    def physical_reads(self):
        return [a for a in self.reads if a.source == 'physical']

    def physical_writes(self):
        return [a for a in self.writes if a.source == 'physical']


@dataclass
class Report:
    """Aggregated report for a whole SQL script."""

    sql: str
    statements: list = field(default_factory=list)

    @property
    def accesses(self):
        return [a for s in self.statements for a in s.accesses]

    @property
    def unknowns(self):
        return [u for s in self.statements for u in s.unknowns]


class _Scope:
    """A query-block scope with the CTE names visible in it."""

    __slots__ = ('_counter', 'ctes', 'label', 'parent')

    def __init__(self, label, ctes=(), parent=None):
        self.label = label
        self.ctes = frozenset(ctes)
        self.parent = parent
        self._counter = 0

    def child(self, infix):
        self._counter += 1
        label = f'{self.label}.{infix}{self._counter}'
        return _Scope(label, parent=self)

    def named_child(self, label):
        return _Scope(f'{self.label}.{label}', parent=self)

    def with_cte(self, name):
        """Return a scope like this one but with ``name`` also visible."""
        return _Scope(self.label, self.ctes | {name.lower()}, self.parent)

    def is_cte(self, name):
        scope = self
        while scope is not None:
            if name.lower() in scope.ctes:
                return True
            scope = scope.parent
        return False


class _Ctx:
    """Per query-block parsing context."""

    __slots__ = ('delete_target_done', 'stmt_type')

    def __init__(self, stmt_type):
        self.stmt_type = stmt_type
        self.delete_target_done = False


class _Analyzer:
    def __init__(self, sql):
        self.sql = sql
        self.offsets = {}
        self.consumed = set()
        self.report = None   # current StatementReport

    # -- offsets ---------------------------------------------------------

    def _index_offsets(self, stmt, base):
        def walk(tl, pos):
            for tok in tl.tokens:
                self.offsets[id(tok)] = (pos, pos + len(tok.value))
                if tok.is_group:
                    walk(tok, pos)
                pos += len(tok.value)
        walk(stmt, base)

    def _span(self, tok):
        return self.offsets[id(tok)]

    # -- name handling ---------------------------------------------------

    def _name_segments(self, identifier):
        """Return the segment tokens of the leading ``a.b.c`` name chain.

        Aliases and anything after the first whitespace/keyword break are
        ignored, so neither aliases nor column qualifiers leak in.
        """
        segments = []
        state = 'name'
        for tok in identifier.tokens:
            if tok.is_whitespace or tok.ttype in T.Comment:
                if segments:
                    break
                continue
            if state == 'name' and tok.ttype in _NAME_TTYPES \
                    and not tok.is_group:
                segments.append(tok)
                state = 'dot'
            elif state == 'dot' and tok.ttype == T.Punctuation \
                    and tok.value == '.':
                state = 'name'
            else:
                break
        return segments

    def _emit_access(self, segments, access, scope):
        start = self._span(segments[0])[0]
        end = self._span(segments[-1])[1]
        raw = self.sql[start:end]
        parts = [_unquote(t.value) for t in segments]
        table = '.'.join(parts)
        # A bare single-segment name may refer to a visible CTE: that is a
        # logical source, never a physical table.
        source = 'physical'
        if len(parts) == 1 and scope.is_cte(parts[0]):
            source = 'cte'
        self.report.accesses.append(TableAccess(
            table=table, raw=raw, access=access, source=source,
            scope=scope.label, span=(start, end)))

    def _unknown(self, reason, tok, scope):
        start, end = self._span(tok)
        self.report.unknowns.append(UnknownConstruct(
            reason=reason, scope=scope.label, span=(start, end),
            text=self.sql[start:end]))

    # -- subqueries --------------------------------------------------------

    @staticmethod
    def _inner_first(paren):
        """First meaningful token inside the parentheses (past the '(')."""
        for tok in paren.tokens:
            if tok.is_whitespace or tok.ttype in T.Comment:
                continue
            if tok.ttype == T.Punctuation and tok.value == '(':
                continue
            return tok
        return None

    def _subquery_paren(self, tok):
        """Return the subquery Parenthesis inside ``tok``, if any."""
        parens = []
        if isinstance(tok, Parenthesis):
            parens = [tok]
        elif isinstance(tok, Identifier):
            parens = [t for t in tok.tokens if isinstance(t, Parenthesis)]
        for par in parens:
            first = self._inner_first(par)
            if first is not None and (
                    first.ttype == T.Keyword.DML
                    or first.ttype == T.Keyword.CTE):
                return par
        return None

    def _process_subquery(self, paren, scope, label=None):
        sub = scope.named_child(label) if label else scope.child('q')
        first = self._inner_first(paren)
        stmt_type = 'SELECT'
        if first is not None and first.ttype == T.Keyword.DML:
            stmt_type = first.normalized
        self._scan_block(paren, sub, _Ctx(stmt_type))

    # -- table references --------------------------------------------------

    def _handle_ref(self, tok, access, scope):
        if isinstance(tok, Identifier):
            par = self._subquery_paren(tok)
            if par is not None:
                self._process_subquery(par, scope)
                return
            for child in tok.tokens:
                if isinstance(child, Function):
                    self._unknown('table-valued function in FROM is not '
                                  'resolved', child, scope)
                    return
            segments = self._name_segments(tok)
            if segments:
                self._emit_access(segments, access, scope)
            else:
                self._unknown('unrecognized table reference', tok, scope)
        elif isinstance(tok, Parenthesis):
            if self._subquery_paren(tok) is not None:
                self._process_subquery(tok, scope)
            elif any(t.ttype in T.Keyword and 'JOIN' in t.normalized
                     for t in tok.tokens):
                # parenthesized join: (a JOIN b ON ...) -- same scope
                self._scan_block(tok, scope, _Ctx('SELECT'))
            else:
                self._unknown('parenthesized table reference is not '
                              'supported', tok, scope)
        elif isinstance(tok, Function):
            self._unknown('table-valued function in FROM is not resolved',
                          tok, scope)
        elif tok.ttype in _NAME_TTYPES and not tok.is_group:
            # Bare name that sqlparse did not wrap into an Identifier
            # (e.g. a table name that collides with a keyword).
            self._emit_access([tok], access, scope)
        elif not tok.is_whitespace and tok.ttype not in T.Comment:
            self._unknown('unrecognized table reference', tok, scope)

    def _consume_refs(self, tokens, i, scope, access, first_access=None):
        """Consume table references starting at tokens[i]."""
        n = len(tokens)
        while i < n and (tokens[i].is_whitespace
                         or tokens[i].ttype in T.Comment):
            i += 1
        if i >= n:
            return i
        tok = tokens[i]
        self.consumed.add(id(tok))
        if isinstance(tok, IdentifierList):
            first = True
            for child in tok.get_identifiers():
                acc = first_access if first and first_access else access
                self._handle_ref(child, acc, scope)
                first = False
        else:
            self._handle_ref(tok, first_access or access, scope)
        return i + 1

    # -- WITH --------------------------------------------------------------

    def _handle_with(self, tokens, i, scope):
        """Process CTE definitions; tokens[i] follows the WITH keyword."""
        n = len(tokens)
        while i < n and (tokens[i].is_whitespace
                         or tokens[i].ttype in T.Comment):
            i += 1
        if i < n and tokens[i].ttype == T.Keyword \
                and tokens[i].normalized == 'RECURSIVE':
            self.consumed.add(id(tokens[i]))
            i += 1
            while i < n and (tokens[i].is_whitespace
                             or tokens[i].ttype in T.Comment):
                i += 1
        if i >= n:
            return i
        defs = tokens[i]
        self.consumed.add(id(defs))
        if isinstance(defs, Identifier):
            defs = [defs]
        elif isinstance(defs, IdentifierList):
            defs = list(defs.get_identifiers())
        else:
            self._unknown('unsupported WITH clause', defs, scope)
            return i + 1
        cur = scope
        for cte in defs:
            name_tok = None
            for tok in cte.tokens:
                if tok.ttype in (T.Name, T.Name.Quoted, T.String.Symbol):
                    name_tok = tok
                    break
            body = self._subquery_paren(cte)
            if name_tok is None or body is None:
                self._unknown('unsupported CTE definition', cte, cur)
                continue
            name = _unquote(name_tok.value)
            # Visible in its own body (a recursive self-reference is then a
            # logical source, never a physical table), in later CTE bodies
            # and in the main query.
            cur = cur.with_cte(name)
            self._process_subquery(body, cur, label=f'cte:{name}')
        # The main query sees every CTE defined here.
        scope.ctes = cur.ctes
        return i + 1

    # -- main block scanner --------------------------------------------------

    def _scan_block(self, block, scope, ctx):
        tokens = block.tokens
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            i += 1
            if id(tok) in self.consumed:
                continue
            if tok.is_whitespace or tok.ttype in T.Comment:
                continue
            if isinstance(tok, Parenthesis):
                if self._subquery_paren(tok) is not None:
                    self._process_subquery(tok, scope)
                continue
            if isinstance(tok, Identifier):
                par = self._subquery_paren(tok)
                if par is not None:
                    self._process_subquery(par, scope)
                continue
            if isinstance(tok, TokenList):
                # Where, Comparison, grouped expressions, ...: scan inside
                # with the same scope; subqueries open their own scope.
                self._scan_block(tok, scope, ctx)
                continue
            if tok.ttype == T.Keyword.CTE and \
                    tok.normalized.startswith('WITH'):
                self.consumed.add(id(tok))
                i = self._handle_with(tokens, i, scope)
                continue
            if tok.ttype not in T.Keyword and tok.ttype != T.Keyword.DML:
                continue
            norm = tok.normalized
            if norm == 'FROM':
                first = None
                if ctx.stmt_type == 'DELETE' and not ctx.delete_target_done:
                    first = 'write'
                    ctx.delete_target_done = True
                i = self._consume_refs(tokens, i, scope, 'read', first)
            elif norm.endswith('JOIN'):
                i = self._consume_refs(tokens, i, scope, 'read')
            elif norm == 'INTO' or (norm == 'UPDATE' and tok.ttype == T.Keyword.DML):
                i = self._consume_refs(tokens, i, scope, 'write')
            elif norm == 'USING' and ctx.stmt_type == 'DELETE':
                i = self._consume_refs(tokens, i, scope, 'read')
            elif norm == 'DELETE' and tok.ttype == T.Keyword.DML:
                # MySQL-style "DELETE t FROM t ...": target before FROM.
                j = i
                while j < n and (tokens[j].is_whitespace
                                 or tokens[j].ttype in T.Comment):
                    j += 1
                if j < n and isinstance(tokens[j], (Identifier,
                                                    IdentifierList)):
                    ctx.delete_target_done = True
                    i = self._consume_refs(tokens, i, scope, 'write')
            else:
                for frag in _UNKNOWN_KEYWORDS:
                    if frag in norm:
                        self._unknown(
                            f'dialect construct {norm!r} is not supported',
                            tok, scope)
                        break

    # -- entry point ---------------------------------------------------------

    def analyze_statement(self, stmt, label, base):
        self._index_offsets(stmt, base)
        stmt_type = stmt.get_type()
        report = StatementReport(label=label, type=stmt_type,
                                 span=(base, base + len(str(stmt))))
        self.report = report
        if stmt_type not in SUPPORTED_TYPES:
            first = stmt.token_first(skip_cm=True)
            self._unknown(
                f'statement type {stmt_type!r} is not supported; no '
                'table access analysis performed',
                first if first is not None else stmt, _Scope(label))
            return report
        scope = _Scope(label)
        self._scan_block(stmt, scope, _Ctx(stmt_type))
        return report


def analyze(sql):
    """Analyze ``sql`` and return a :class:`Report` (offline, no database)."""
    report = Report(sql=sql)
    analyzer = _Analyzer(sql)
    cursor = 0
    for idx, stmt in enumerate(sqlparse.parse(sql), start=1):
        text = str(stmt)
        base = sql.find(text, cursor)
        if base < 0:
            base = cursor
        cursor = base + len(text)
        label = f'S{idx}'
        report.statements.append(analyzer.analyze_statement(stmt, label, base))
    return report


def format_report(report):
    """Render a :class:`Report` using the attribution line format.

    One header per statement, then one line per table access::

        <scope> | <access> | <source> | <table> | span=<a>:<b> | raw=<raw>

    followed by one ``UNKNOWN`` line per unsupported construct.
    """
    lines = []
    for stmt in report.statements:
        lines.append(f'== {stmt.label} {stmt.type} '
                     f'span={stmt.span[0]}:{stmt.span[1]} ==')
        for acc in stmt.accesses:
            lines.append(f'{acc.scope} | {acc.access} | {acc.source} | '
                         f'{acc.table} | span={acc.span[0]}:{acc.span[1]}'
                         f' | raw={acc.raw}')
        for unk in stmt.unknowns:
            lines.append(f'{unk.scope} | UNKNOWN | {unk.reason} | '
                         f'span={unk.span[0]}:{unk.span[1]} | '
                         f'text={unk.text}')
    return '\n'.join(lines)


_DEMO_SQL = [
    # 1. CTE + JOIN: physical reads must be exactly {raw, target}.
    'WITH t AS (SELECT * FROM raw) '
    'SELECT x.id FROM t x JOIN target y ON x.id = y.id',
    # 2. Write vs read, quoted names, nested subquery.
    'INSERT INTO "mart"."Daily Stats" (id) '
    'SELECT o.id FROM staging.orders o '
    'WHERE o.id NOT IN (SELECT id FROM ops.blacklist)',
    # 3. Unsupported dialect construct: reported as unknown, no full parse.
    'MERGE INTO tgt USING src ON tgt.id = src.id '
    'WHEN MATCHED THEN UPDATE SET x = 1',
]


def main():
    for sql in _DEMO_SQL:
        print(sql)
        print(format_report(analyze(sql)))
        print()


if __name__ == '__main__':
    main()
