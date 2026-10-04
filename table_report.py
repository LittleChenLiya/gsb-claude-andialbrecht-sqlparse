"""离线表访问报告：基于 sqlparse 的静态 SQL 表访问分析。

本工具不连接数据库、不执行 SQL，仅通过 sqlparse 的语法树静态识别每条语句中
实际读取/写入的物理表、CTE 逻辑来源、原文区间以及对应的子查询作用域。

支持范围（明确不声称完整解析）：
- 语句：SELECT（含 SELECT INTO）、INSERT（含 REPLACE）、UPDATE、DELETE；
- 结构：WITH（含 RECURSIVE）、JOIN、嵌套子查询、UNION/EXCEPT/INTERSECT、
  UPDATE ... FROM、DELETE ... USING、INSERT ... SELECT、派生表别名与列别名表；
- CTE 名只在其可见作用域内作为逻辑来源；递归 CTE 自引用不会被误报成物理表；
- 列限定符（如 x.id）与别名（如 t x）不会被当成额外表名；
- schema 限定（dbo.target）与引号标识符（"dbo"."target"、[dbo].[target]）
  保留真实分段；
- 不支持的方言结构（CREATE/MERGE/GRANT/表函数等）明确报为 unknown。

名称解析按大小写不敏感处理（静态分析无法核对数据库中引号标识符的大小写）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sqlparse
import sqlparse.sql as sql
from sqlparse import tokens as T

__all__ = [
    "TableRef", "Unknown", "StatementReport",
    "analyze_sql", "render_report", "render_summary", "demo",
]


@dataclass
class CteDef:
    name: str
    body: sql.Parenthesis
    recursive: bool
    scope_id: str


@dataclass
class Scope:
    sid: str
    kind: str  # root | subquery | cte
    parent: Optional["Scope"]
    ctes: Dict[str, CteDef]
    range: Optional[Tuple[int, int]] = None

    def child(self, kind: str, sid: str,
              cte_overrides: Optional[Dict[str, CteDef]] = None) -> "Scope":
        ctes = dict(self.ctes)
        if cte_overrides:
            ctes.update(cte_overrides)
        return Scope(sid=sid, kind=kind, parent=self, ctes=ctes)


@dataclass
class TableRef:
    role: str            # read | write
    segments: List[str]  # 真实分段，如 ['dbo', 'target']
    alias: Optional[str]
    source: str          # physical | cte
    cte_name: Optional[str]
    range: Tuple[int, int]
    scope: str
    text: str

    @property
    def qualified(self) -> str:
        return ".".join(self.segments)


@dataclass
class Unknown:
    kind: str
    reason: str
    range: Tuple[int, int]
    text: str


@dataclass
class StatementReport:
    index: int
    stmt_type: str       # SELECT | INSERT | UPDATE | DELETE | UNKNOWN
    range: Tuple[int, int]
    sql: str
    reads: List[TableRef] = field(default_factory=list)
    writes: List[TableRef] = field(default_factory=list)
    unknowns: List[Unknown] = field(default_factory=list)
    scopes: List[dict] = field(default_factory=list)

    def physical_set(self, role: str) -> List[str]:
        refs = self.reads if role == "read" else self.writes
        return sorted({r.qualified for r in refs if r.source == "physical"})


_JOIN_PREFIXES = {
    "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL",
    "ASYMMETRIC", "SYMMETRIC",
}
_CLAUSE_KEYWORDS = {
    "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET", "FETCH",
    "FOR", "RETURNING", "WINDOW", "QUALIFY", "PIVOT", "UNPIVOT",
    "MATCH_RECOGNIZE",
}
_SET_OPS = {"UNION", "EXCEPT", "INTERSECT"}
_DML_STARTS = {"SELECT", "INSERT", "UPDATE", "DELETE", "VALUES"}
# 不会被当作表名的关键字（结构关键字 / 表达式关键字）。
# sqlparse 会把各方言关键字表中的词（如 archive、raw）统一标成 Keyword，
# 因此在表因子位置遇到 Keyword 时，需要用这张黑名单排除结构/表达式用法。
_NON_TABLE_KEYWORDS = {
    "AS", "ONLY", "LATERAL", "USING", "ON", "DISTINCT", "ALL", "AND", "OR",
    "NOT", "IN", "IS", "LIKE", "ILIKE", "RLIKE", "REGEXP", "BETWEEN",
    "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END", "OVER", "PARTITION",
    "ROWS", "RANGE", "GROUPS", "FILTER", "WITHIN", "ASC", "DESC", "NULLS",
    "FIRST", "LAST", "TABLESAMPLE", "SYSTEM", "NULL", "TRUE", "FALSE",
    "UNKNOWN", "DO", "NOTHING", "CONFLICT", "DUPLICATE", "KEY", "IGNORE",
    "PRIOR", "CONNECT", "START", "MATCH", "MEASURES", "PATTERN", "DEFINE",
    "RUNNING", "FINAL", "KEEP", "EXCLUDE", "TIES", "OTHERS", "CAST",
    "COALESCE", "NULLIF", "GREATEST", "LEAST", "OF", "TIMESTAMP", "TIME",
    "DATE", "DATETIME", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND",
    "SYSTEM_TIME", "FOR", "SET", "BY", "GROUP", "HAVING", "WHERE", "ORDER",
    "LIMIT", "OFFSET", "FETCH", "RETURNING", "WINDOW", "QUALIFY", "PIVOT",
    "UNPIVOT", "MATCH_RECOGNIZE", "UNION", "EXCEPT", "INTERSECT", "SELECT",
    "INSERT", "UPDATE", "DELETE", "VALUES", "WITH", "REPLACE", "RECURSIVE",
    "EXPLAIN", "INTO", "FROM", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
    "OUTER", "CROSS", "NATURAL", "ASYMMETRIC", "SYMMETRIC", "CREATE",
    "ALTER", "DROP", "GRANT", "REVOKE", "TRUNCATE", "MERGE", "COPY", "CALL",
    "EXEC", "USE", "SHOW", "DESCRIBE", "PRAGMA", "VACUUM", "HANDLER",
    "LOAD", "LOCK", "UNLOCK", "ATTACH", "DETACH", "REINDEX", "CHECKPOINT",
    "SAVEPOINT", "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION", "DECLARE",
    "OPEN", "CLOSE", "MOVE", "GET", "DIAGNOSTICS", "SIGNAL", "RESIGNAL",
    "WHILE", "LOOP", "REPEAT", "LEAVE", "ITERATE", "RETURN", "IF",
    "ELSIF", "ELSEIF", "TABLE", "INDEX", "COLUMN", "DATABASE", "SCHEMA",
    "VIEW", "MATERIALIZED", "SEQUENCE", "FUNCTION", "PROCEDURE", "TRIGGER",
    "ROLE", "CONSTRAINT", "PRIMARY", "FOREIGN", "REFERENCES", "CHECK",
    "DEFAULT", "COLLATE", "GENERATED", "ALWAYS", "STORED", "VIRTUAL",
    "ZONE", "HOUR_MICROSECOND",
}


def _is_meaningful(tok) -> bool:
    if tok is None:
        return False
    if tok.ttype in (T.Whitespace, T.Comment):
        return False
    if isinstance(tok, sql.Comment):
        return False
    return True


def _first_meaningful(tokens):
    for tok in tokens:
        if _is_meaningful(tok):
            return tok
    return None


def _is_join_keyword(tok) -> bool:
    if tok is None or tok.ttype not in T.Keyword:
        return False
    u = tok.value.upper()
    return u == "JOIN" or u.endswith(" JOIN")


def _first_stmt_keyword(tokens):
    """找到括号/语句块内第一个 DML 或 WITH 关键字（跳过 '(' 等标点）。"""
    for tok in tokens:
        if tok.ttype in T.Keyword:
            u = tok.value.upper()
            if u in _DML_STARTS or u == "WITH":
                return tok
    return None


def _is_join_group_paren(paren: sql.Parenthesis) -> bool:
    """判断 FROM 模式下的括号是否为 (a JOIN b) / (a, b) 连接组。

    其它括号（USING (id) 列清单、ON 中的 (a.id = b.id) 条件等）不是。
    """
    saw_comma = False
    saw_ident = False
    for tok in paren.tokens:
        if not _is_meaningful(tok):
            continue
        if _is_join_keyword(tok):
            return True
        if tok.ttype in T.Operator:
            return False
        if tok.ttype in T.Keyword:
            return False
        if isinstance(tok, sql.IdentifierList):
            saw_comma = True
            for c in tok.tokens:
                if isinstance(c, sql.Identifier):
                    saw_ident = True
            continue
        if isinstance(tok, sql.Identifier):
            saw_ident = True
            continue
        if tok.ttype == T.Punctuation and tok.value == ",":
            saw_comma = True
            continue
        return False
    return saw_ident and saw_comma


def _unquote(value: str) -> str:
    """去掉引号/方括号并还原转义，保留真实分段名。"""
    if len(value) >= 2:
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1].replace('""', '"')
        if value.startswith("`") and value.endswith("`"):
            return value[1:-1].replace("``", "`")
        if value.startswith("[") and value.endswith("]"):
            return value[1:-1]
    return value


def _extract_segments(ident: sql.Identifier) -> List[str]:
    """从 Identifier 中抽取真实分段（去掉别名、列清单与引号）。"""
    segs: List[str] = []
    for tok in ident.tokens:
        if not _is_meaningful(tok):
            continue
        if isinstance(tok, sql.Function):
            # 形如 dbo.target (id) 或 CTE 名 a(x)：Function 名是表名/CTE 名，
            # 括号是列清单或参数
            fname = _function_name(tok)
            if fname:
                segs.append(fname)
            break
        if isinstance(tok, (sql.Parenthesis, sql.Identifier)):
            break  # 列清单 / 别名
        if tok.ttype == T.Punctuation and tok.value == ".":
            continue
        if tok.ttype in T.Keyword:
            if tok.value.upper() == "AS":
                break
            continue
        if tok.ttype == T.Wildcard:
            break
        val = tok.value
        if tok.ttype == T.Literal.String.Symbol:
            val = _unquote(val)
        elif tok.ttype in T.Name:
            val = _unquote(val)
        else:
            continue
        segs.append(val)
    return segs


def _extract_alias(ident: sql.Identifier) -> Optional[str]:
    alias = ident.get_alias()
    if alias:
        return _unquote(alias)
    return None


def _function_name(func: sql.Function) -> Optional[str]:
    for tok in func.tokens:
        if isinstance(tok, sql.Identifier):
            segs = _extract_segments(tok)
            if segs:
                return segs[0]
        if tok.ttype in T.Name:
            return _unquote(tok.value)
    return None


def _build_offsets(tokens, start: int = 0, out: Optional[dict] = None) -> dict:
    """递归计算每个 token 在原文中的 [起, 止) 区间。"""
    if out is None:
        out = {}
    for tok in tokens:
        end = start + len(tok.value)
        out[id(tok)] = (start, end)
        if tok.is_group:
            _build_offsets(tok.tokens, start, out)
        start = end
    return out


class Analyzer:
    def __init__(self, sql_text: str):
        self.sql_text = sql_text
        self.offsets: dict = {}
        self._scope_seq = 0
        self._subq_seq = 0
        self._last_ref: Optional[TableRef] = None

    def _new_scope(self, kind: str, parent: Scope, rng) -> Scope:
        if kind == "root":
            sid = "root"
        elif kind == "cte":
            self._scope_seq += 1
            sid = "cte#%d" % self._scope_seq
        else:
            self._subq_seq += 1
            sid = "subquery#%d" % self._subq_seq
        scope = Scope(sid=sid, kind=kind, parent=parent,
                      ctes=dict(parent.ctes), range=rng)
        return scope

    def analyze(self, stmt: sql.Statement, index: int) -> StatementReport:
        self.offsets = _build_offsets(stmt.tokens)
        rng = (0, len(stmt.value))
        self.offsets[id(stmt)] = rng
        rep = StatementReport(index=index, stmt_type="UNKNOWN",
                              range=rng, sql=stmt.value)
        root = self._new_scope("root", Scope("root", None, None, {}), rng)
        rep.scopes.append({"id": root.sid, "kind": "root",
                           "range": rng, "ctes": []})

        top = list(stmt.tokens)
        first = _first_meaningful(top)
        if first is not None and first.ttype in T.Keyword \
                and first.value.upper() == "EXPLAIN":
            first = _first_meaningful([t for t in top if t is not first])

        if first is not None and first.ttype in T.Keyword \
                and first.value.upper() == "WITH":
            self._handle_with(top, root, rep)
            return rep

        kw = first.value.upper() if (first is not None
                                     and first.ttype in T.Keyword) else None
        if kw in _DML_STARTS:
            rep.stmt_type = {"SELECT": "SELECT", "INSERT": "INSERT",
                             "UPDATE": "UPDATE", "DELETE": "DELETE",
                             "VALUES": "SELECT"}[kw]
            self._walk(top, root, rep, mode="init")
        elif kw is not None:
            rep.unknowns.append(Unknown(
                kind="unsupported_statement",
                reason="不支持的语句类型 %r，仅支持 SELECT/INSERT/UPDATE/DELETE/WITH" % kw,
                range=rng, text=stmt.value.strip(),
            ))
        else:
            rep.unknowns.append(Unknown(
                kind="unresolved_statement",
                reason="无法识别的语句结构",
                range=rng, text=stmt.value.strip(),
            ))
        return rep

    def _handle_with(self, tokens, scope: Scope, rep: StatementReport) -> None:
        recursive = False
        cte_list_tok = None
        seen_with = False
        for tok in tokens:
            if not _is_meaningful(tok):
                continue
            if tok.ttype in T.Keyword:
                u = tok.value.upper()
                if u == "WITH" and not seen_with:
                    seen_with = True
                    continue
                if u == "RECURSIVE":
                    recursive = True
                    continue
                if u == "EXPLAIN":
                    continue
            cte_list_tok = tok
            break

        cte_idents: List[sql.Identifier] = []
        if isinstance(cte_list_tok, sql.IdentifierList):
            for c in cte_list_tok.tokens:
                if isinstance(c, sql.Identifier):
                    cte_idents.append(c)
        elif isinstance(cte_list_tok, sql.Identifier):
            cte_idents.append(cte_list_tok)
        else:
            rep.unknowns.append(Unknown(
                kind="unresolved_cte",
                reason="WITH 后未找到 CTE 定义清单",
                range=self.offsets.get(id(cte_list_tok), (0, 0)),
                text=cte_list_tok.value if cte_list_tok is not None else "",
            ))
            self._walk(tokens, scope, rep, mode="init")
            return

        for ident in cte_idents:
            segs = _extract_segments(ident)
            if not segs:
                rep.unknowns.append(Unknown(
                    kind="unresolved_cte",
                    reason="无法识别的 CTE 名称",
                    range=self.offsets[id(ident)], text=ident.value,
                ))
                continue
            cte_name = segs[0]
            as_idx = None
            for i, t in enumerate(ident.tokens):
                if t.ttype in T.Keyword and t.value.upper() == "AS":
                    as_idx = i
                    break
            body = None
            if as_idx is not None:
                for t in ident.tokens[as_idx + 1:]:
                    if isinstance(t, sql.Parenthesis):
                        body = t
                        break
            if body is None:
                rep.unknowns.append(Unknown(
                    kind="unresolved_cte",
                    reason="CTE %r 缺少 AS (定义体)" % cte_name,
                    range=self.offsets[id(ident)], text=ident.value,
                ))
                continue

            child = self._new_scope("cte", scope, self.offsets[id(body)])
            cte_def = CteDef(name=cte_name, body=body,
                             recursive=recursive, scope_id=child.sid)
            if recursive:
                child.ctes[cte_name.lower()] = cte_def
            rep.scopes.append({
                "id": child.sid, "kind": "cte",
                "range": self.offsets[id(body)], "ctes": [cte_name],
            })
            self._analyze_block(body.tokens, child, rep)
            scope.ctes[cte_name.lower()] = cte_def

        main_kw = None
        for tok in tokens:
            if tok.ttype in T.Keyword and tok.value.upper() in _DML_STARTS:
                main_kw = tok.value.upper()
                break
        if main_kw is None:
            rep.unknowns.append(Unknown(
                kind="unresolved_statement",
                reason="WITH 之后未找到主语句",
                range=rep.range, text="",
            ))
            return
        rep.stmt_type = {"SELECT": "SELECT", "INSERT": "INSERT",
                         "UPDATE": "UPDATE", "DELETE": "DELETE",
                         "VALUES": "SELECT"}[main_kw]
        self._walk(tokens, scope, rep, mode="init")

    def _analyze_block(self, tokens, scope: Scope,
                       rep: StatementReport) -> None:
        first = _first_stmt_keyword(tokens)
        if first is not None and first.value.upper() == "WITH":
            self._handle_with(tokens, scope, rep)
            return
        kw = first.value.upper() if first is not None else None
        if kw not in _DML_STARTS:
            rep.unknowns.append(Unknown(
                kind="unresolved_subquery",
                reason="子查询内未找到支持的 DML 语句（首关键字 %r）" % kw,
                range=self.offsets.get(id(first), (0, 0)),
                text=first.value if first is not None else "",
            ))
            return
        self._walk(tokens, scope, rep, mode="init")

    def _walk(self, tokens, scope, rep, mode):
        union_next = False
        last_was_subquery = False
        skip_next_paren = False
        expect_alias = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not _is_meaningful(tok):
                i += 1
                continue
            if isinstance(tok, sql.Parenthesis):
                inner = _first_stmt_keyword(tok.tokens)
                if inner is not None:
                    child = self._new_scope("subquery", scope, self.offsets[id(tok)])
                    rep.scopes.append({"id": child.sid, "kind": "subquery",
                                       "range": self.offsets[id(tok)],
                                       "ctes": list(child.ctes)})
                    self._analyze_block(tok.tokens, child, rep)
                    last_was_subquery = True
                elif mode == "from" and not skip_next_paren \
                        and _is_join_group_paren(tok):
                    self._walk(tok.tokens, scope, rep, mode="from")
                    last_was_subquery = False
                else:
                    last_was_subquery = False
                skip_next_paren = False
                expect_alias = False
                i += 1
                continue
            if isinstance(tok, (sql.Identifier, sql.IdentifierList, sql.Function)):
                if mode in ("from", "insert_into", "update_target",
                            "delete_before", "delete_from", "select_into"):
                    if mode == "from" and expect_alias \
                            and isinstance(tok, sql.Identifier):
                        # 紧跟在表名后的别名（如 raw r），不是表
                        alias = _extract_alias(tok)
                        if alias is None:
                            segs = _extract_segments(tok)
                            alias = segs[0] if segs else None
                        if alias and self._last_ref is not None:
                            self._last_ref.alias = alias
                        expect_alias = False
                        i += 1
                        continue
                    role = "write" if mode in ("insert_into", "update_target",
                                               "delete_before", "delete_from",
                                               "select_into") else "read"
                    sub = self._process_factor(tok, scope, rep, role,
                                               last_was_subquery=last_was_subquery)
                    last_was_subquery = (sub == "subquery")
                    expect_alias = (sub == "factor")
                    if mode == "insert_into":
                        mode = "select"
                    elif mode == "delete_from":
                        mode = "clause"
                    elif mode == "select_into":
                        mode = "clause"
                else:
                    # 非因子模式下，Identifier 可能是 SELECT 列表中的子查询
                    if isinstance(tok, sql.Identifier):
                        first = _first_meaningful(tok.tokens)
                        if isinstance(first, sql.Parenthesis) \
                                and _first_stmt_keyword(first.tokens) is not None:
                            child = self._new_scope("subquery", scope,
                                                    self.offsets[id(first)])
                            rep.scopes.append({
                                "id": child.sid, "kind": "subquery",
                                "range": self.offsets[id(first)],
                                "ctes": list(child.ctes),
                            })
                            self._analyze_block(first.tokens, child, rep)
                    last_was_subquery = False
                    expect_alias = False
                i += 1
                continue
            if tok.is_group:
                self._walk(tok.tokens, scope, rep, mode="clause")
                last_was_subquery = False
                expect_alias = False
                i += 1
                continue
            if tok.ttype in T.Keyword:
                u = tok.value.upper()
                if u in _SET_OPS or u.split()[0] in _SET_OPS:
                    union_next = True
                    mode = "clause"
                    expect_alias = False
                elif u in _CLAUSE_KEYWORDS:
                    mode = "clause"
                    expect_alias = False
                elif u == "FROM":
                    if mode in ("select", "clause", "from", "select_into"):
                        mode = "from"
                    elif mode in ("update_target", "update_set"):
                        mode = "from"
                    elif mode == "delete":
                        mode = "delete_from"
                    expect_alias = False
                elif _is_join_keyword(tok):
                    mode = "from"
                    expect_alias = False
                elif u in _JOIN_PREFIXES:
                    expect_alias = False
                elif u == "AS" and mode == "from":
                    expect_alias = True
                elif u in ("SELECT", "VALUES"):
                    if mode in ("init", "insert", "insert_into", "from") \
                            or union_next:
                        mode = "select"
                        union_next = False
                    expect_alias = False
                elif u == "INSERT":
                    if mode == "init" or union_next:
                        mode = "insert"
                        union_next = False
                    expect_alias = False
                elif u == "UPDATE":
                    if mode == "init" or union_next:
                        mode = "update_target"
                        union_next = False
                    expect_alias = False
                elif u == "DELETE":
                    if mode == "init" or union_next:
                        mode = "delete"
                        union_next = False
                    expect_alias = False
                elif u == "REPLACE":
                    if mode == "init" or union_next:
                        mode = "insert"
                        union_next = False
                    expect_alias = False
                elif u == "INTO":
                    if mode == "insert":
                        mode = "insert_into"
                    elif mode in ("select", "clause"):
                        mode = "select_into"
                    expect_alias = False
                elif u == "SET":
                    if mode in ("update_target", "update_set"):
                        mode = "update_set"
                    expect_alias = False
                elif u == "USING":
                    if mode in ("delete", "clause"):
                        mode = "from"  # DELETE ... USING：读来源
                    elif mode == "from":
                        skip_next_paren = True  # JOIN ... USING (id) 列清单
                    expect_alias = False
                elif u == "EXPLAIN":
                    expect_alias = False
                elif u == "WITH":
                    expect_alias = False
                elif mode in ("from", "insert_into", "update_target",
                              "delete_before", "delete_from", "select_into") \
                        and u not in _NON_TABLE_KEYWORDS:
                    # 因子位置上的表名（sqlparse 可能将其标为 Keyword）
                    role = "write" if mode in (
                        "insert_into", "update_target", "delete_before",
                        "delete_from", "select_into") else "read"
                    sub = self._process_factor(tok, scope, rep, role,
                                               last_was_subquery)
                    expect_alias = (sub == "factor")
            i += 1

    def _process_factor(self, tok, scope, rep, role, last_was_subquery):
        if isinstance(tok, sql.IdentifierList):
            for c in tok.tokens:
                if isinstance(c, sql.Identifier):
                    self._process_factor(c, scope, rep, role, last_was_subquery)
            return "factor"

        if tok.ttype in T.Keyword:
            u = tok.value.upper()
            if u in _NON_TABLE_KEYWORDS:
                return "skip"
            return self._add_ref([tok.value], None, tok, scope, rep, role)

        if isinstance(tok, sql.Function):
            if last_was_subquery:
                return "subquery"
            name = _function_name(tok)
            if role == "write" and name:
                # INSERT INTO t (col) 等：函数名位置是目标表，括号是列清单
                return self._add_ref([name], None, tok, scope, rep, role)
            rep.unknowns.append(Unknown(
                kind="table_function",
                reason="表函数 %r 不是物理表，无法进一步静态解析" % name,
                range=self.offsets[id(tok)], text=tok.value,
            ))
            return "skip"

        if isinstance(tok, sql.Identifier):
            first = _first_meaningful(tok.tokens)
            if isinstance(first, sql.Parenthesis):
                inner = _first_stmt_keyword(first.tokens)
                if inner is not None:
                    child = self._new_scope("subquery", scope, self.offsets[id(first)])
                    rep.scopes.append({
                        "id": child.sid, "kind": "subquery",
                        "range": self.offsets[id(first)],
                        "ctes": list(child.ctes),
                    })
                    self._analyze_block(first.tokens, child, rep)
                    return "subquery"
                self._walk(first.tokens, scope, rep, mode="from")
                return "factor"

            if isinstance(first, sql.Function):
                # 表函数（如 generate_series(1,10) g），不是物理表
                rep.unknowns.append(Unknown(
                    kind="table_function",
                    reason="表函数 %r 不是物理表，无法进一步静态解析"
                    % _function_name(first),
                    range=self.offsets[id(tok)], text=tok.value,
                ))
                return "skip"

            segs = _extract_segments(tok)
            if not segs:
                if any(isinstance(c, sql.Function) for c in tok.tokens):
                    rep.unknowns.append(Unknown(
                        kind="table_function",
                        reason="表函数不是物理表，无法进一步静态解析",
                        range=self.offsets[id(tok)], text=tok.value,
                    ))
                    return "skip"
                rep.unknowns.append(Unknown(
                    kind="unresolved_factor",
                    reason="无法解析的表因子",
                    range=self.offsets[id(tok)], text=tok.value,
                ))
                return "skip"
            alias = _extract_alias(tok)
            return self._add_ref(segs, alias, tok, scope, rep, role)

        return "skip"

    def _add_ref(self, segs, alias, tok, scope, rep, role):
        name_lower = segs[-1].lower()
        cte = scope.ctes.get(name_lower)
        if cte is not None:
            ref = TableRef(
                role=role, segments=segs, alias=alias,
                source="cte", cte_name=cte.name,
                range=self.offsets[id(tok)], scope=scope.sid, text=tok.value,
            )
        else:
            ref = TableRef(
                role=role, segments=segs, alias=alias,
                source="physical", cte_name=None,
                range=self.offsets[id(tok)], scope=scope.sid, text=tok.value,
            )
        if role == "write":
            rep.writes.append(ref)
        else:
            rep.reads.append(ref)
        self._last_ref = ref
        return "factor"


def analyze_sql(sql_text: str) -> List[StatementReport]:
    """离线分析一到多条 SQL，返回每条语句的表访问报告。"""
    reports: List[StatementReport] = []
    analyzer = Analyzer(sql_text)
    for i, stmt in enumerate(sqlparse.parse(sql_text)):
        if not _first_meaningful(stmt.tokens):
            continue
        reports.append(analyzer.analyze(stmt, i))
    return reports


def _render_ref(ref: TableRef) -> List[str]:
    tag = "物理" if ref.source == "physical" else "CTE逻辑"
    role = "写入" if ref.role == "write" else "读取"
    alias = ref.alias if ref.alias else "无"
    lines = [
        "  [%s%s] 表名: %s  分段: %r  别名: %s  归属作用域: %s  区间: [%d, %d)  原文: %r"
        % (tag, role, ref.qualified, ref.segments, alias,
           ref.scope, ref.range[0], ref.range[1], ref.text),
    ]
    if ref.source == "cte":
        lines.append("           └─ 作为 CTE %r 的逻辑来源（非物理表）" % ref.cte_name)
    return lines


def render_report(rep: StatementReport) -> str:
    lines = [
        "语句 #%d  类型: %s  区间: [%d, %d)"
        % (rep.index, rep.stmt_type, rep.range[0], rep.range[1]),
        "SQL: %r" % rep.sql.strip(),
    ]
    if rep.writes:
        lines.append("  写入的表:")
        for r in rep.writes:
            lines.extend(_render_ref(r))
    if rep.reads:
        lines.append("  读取的表:")
        for r in rep.reads:
            lines.extend(_render_ref(r))
    if rep.unknowns:
        lines.append("  未知/不支持的结构（不声称完整解析）:")
        for u in rep.unknowns:
            lines.append("  [未知:%s] %s  区间: [%d, %d)  原文: %r"
                         % (u.kind, u.reason, u.range[0], u.range[1], u.text))
    lines.append("  作用域: %r" % [s["id"] for s in rep.scopes])
    lines.append("  物理读取集合: %r" % rep.physical_set("read"))
    lines.append("  物理写入集合: %r" % rep.physical_set("write"))
    return "\n".join(lines)


def render_summary(reports: List[StatementReport]) -> str:
    out = []
    for rep in reports:
        out.append(render_report(rep))
        out.append("")
    return "\n".join(out)


_DEMOS = [
    """-- 演示 1：CTE + JOIN，CTE 名只作逻辑来源
WITH t AS (SELECT * FROM raw)
SELECT x.id FROM t x JOIN target y ON x.id = y.id""",
    """-- 演示 2：内层 CTE 遮蔽同名外部表，写入与 SELECT 来源分开标记
WITH t AS (SELECT * FROM t)
INSERT INTO archive (id)
SELECT id FROM t""",
    """-- 演示 3：引号/schema 限定与嵌套子查询
INSERT INTO dbo."Order" (id)
SELECT o.id FROM dbo."Order" o
WHERE o.id IN (SELECT id FROM "Raw" r JOIN ref.schema_item i ON r.id = i.raw_id)""",
]


def demo() -> str:
    chunks = []
    for i, sql_text in enumerate(_DEMOS, 1):
        chunks.append("===== 演示 %d =====" % i)
        chunks.append(sql_text)
        chunks.append("")
        reports = analyze_sql(sql_text)
        chunks.append(render_summary(reports))
    return "\n".join(chunks)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(demo())
