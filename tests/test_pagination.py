#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Tests for the explicit pagination rewrite."""

import pytest

import sqlparse
from sqlparse.pagination import (
    PaginationError,
    PaginationPlan,
    PaginationValue,
    UnsupportedDialectError,
    UnsupportedStatementError,
    parse_pagination,
    rewrite_pagination,
)


class TestRewritePagination:
    """Tests for rewrite_pagination."""

    def test_add_limit_to_select_without_pagination(self):
        result = rewrite_pagination('SELECT * FROM t', limit=10)
        assert result == 'SELECT * FROM t LIMIT 10'

    def test_add_limit_and_offset(self):
        result = rewrite_pagination('SELECT * FROM t', limit=10, offset=5)
        assert result == 'SELECT * FROM t LIMIT 10 OFFSET 5'

    def test_replace_existing_limit(self):
        result = rewrite_pagination('SELECT * FROM t LIMIT 10', limit=20)
        assert result == 'SELECT * FROM t LIMIT 20'

    def test_replace_existing_offset(self):
        result = rewrite_pagination(
            'SELECT * FROM t LIMIT 10 OFFSET 5', offset=3)
        assert result == 'SELECT * FROM t LIMIT 10 OFFSET 3'

    def test_replace_both(self):
        result = rewrite_pagination(
            'SELECT * FROM t LIMIT 10 OFFSET 5', limit=20, offset=3)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET 3'

    def test_add_offset_to_existing_limit(self):
        result = rewrite_pagination('SELECT * FROM t LIMIT 10', offset=5)
        assert result == 'SELECT * FROM t LIMIT 10 OFFSET 5'

    def test_no_args_returns_unchanged(self):
        result = rewrite_pagination('SELECT * FROM t')
        assert result == 'SELECT * FROM t'

    def test_keep_existing_limit_when_only_offset_given(self):
        result = rewrite_pagination('SELECT * FROM t LIMIT 10', offset=5)
        assert result == 'SELECT * FROM t LIMIT 10 OFFSET 5'

    def test_keep_existing_offset_when_only_limit_given(self):
        result = rewrite_pagination(
            'SELECT * FROM t LIMIT 10 OFFSET 5', limit=20)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET 5'


class TestSubqueryAndCte:
    """Subqueries and CTEs must not be affected."""

    def test_subquery_limit_unchanged(self):
        sql = 'SELECT * FROM (SELECT a FROM t LIMIT 5) x'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 10'
        assert 'LIMIT 5' in result

    def test_subquery_limit_replace_outer(self):
        sql = 'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 10'
        result = rewrite_pagination(sql, limit=20)
        assert result == 'SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 20'

    def test_cte_subquery_limit_unchanged(self):
        sql = 'WITH cte AS (SELECT a FROM t LIMIT 5) SELECT * FROM cte'
        result = rewrite_pagination(sql, limit=10)
        assert result == (
            'WITH cte AS (SELECT a FROM t LIMIT 5) SELECT * FROM cte LIMIT 10')

    def test_nested_subquery_limit(self):
        sql = 'SELECT * FROM (SELECT a FROM (SELECT b FROM t LIMIT 3) y) x'
        result = rewrite_pagination(sql, limit=10)
        assert result == (
            'SELECT * FROM (SELECT a FROM (SELECT b FROM t LIMIT 3) y) x '
            'LIMIT 10')


class TestCommentsAndStrings:
    """LIMIT in comments and strings must not participate."""

    def test_limit_in_line_comment(self):
        sql = 'SELECT * FROM t -- LIMIT 999'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM t LIMIT 10 -- LIMIT 999'

    def test_limit_in_block_comment(self):
        sql = 'SELECT * FROM t /* LIMIT 999 */'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM t LIMIT 10 /* LIMIT 999 */'

    def test_limit_in_string(self):
        sql = "SELECT * FROM t WHERE x = 'LIMIT 999'"
        result = rewrite_pagination(sql, limit=10)
        assert result == "SELECT * FROM t WHERE x = 'LIMIT 999' LIMIT 10"

    def test_comment_between_limit_and_value(self):
        sql = 'SELECT * FROM t LIMIT /* c */ 10'
        plan = parse_pagination(sql)
        assert plan.limit.value == '10'
        result = rewrite_pagination(sql, limit=20)
        assert result == 'SELECT * FROM t LIMIT 20'


class TestPlaceholders:
    """Parameterized pagination keeps placeholder identity."""

    def test_question_placeholder_kept(self):
        sql = 'SELECT * FROM t LIMIT ?'
        result = rewrite_pagination(sql, offset=5)
        assert result == 'SELECT * FROM t LIMIT ? OFFSET 5'

    def test_named_placeholder_kept(self):
        sql = 'SELECT * FROM t LIMIT :lim OFFSET :off'
        result = rewrite_pagination(sql, limit=20)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET :off'

    def test_percent_placeholder_kept(self):
        sql = 'SELECT * FROM t LIMIT %s'
        result = rewrite_pagination(sql, offset=5)
        assert result == 'SELECT * FROM t LIMIT %s OFFSET 5'

    def test_dollar_placeholder_kept(self):
        sql = 'SELECT * FROM t LIMIT $1 OFFSET $2'
        result = rewrite_pagination(sql, limit=20)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET $2'

    def test_placeholder_value_parsed(self):
        plan = parse_pagination('SELECT * FROM t LIMIT ? OFFSET :off')
        assert plan.limit == PaginationValue('?', True)
        assert plan.offset == PaginationValue(':off', True)

    def test_placeholder_as_new_value(self):
        result = rewrite_pagination('SELECT * FROM t', limit=':lim')
        assert result == 'SELECT * FROM t LIMIT :lim'
        plan = parse_pagination(result)
        assert plan.limit.is_placeholder is True

    def test_question_as_new_value(self):
        result = rewrite_pagination('SELECT * FROM t', limit='?')
        assert result == 'SELECT * FROM t LIMIT ?'


class TestBytePreservation:
    """Unmodified regions keep their exact original bytes."""

    def test_bytes_outside_span_preserved(self):
        sql = 'SELECT a, b FROM t WHERE x = 1 LIMIT 10 OFFSET 5'
        plan = parse_pagination(sql)
        start, end = plan.span
        result = rewrite_pagination(sql, limit=20, offset=3)
        new_clause = 'LIMIT 20 OFFSET 3'
        assert result == sql[:start] + new_clause + sql[end:]
        assert result[:start] == sql[:start]
        assert result[start + len(new_clause):] == sql[end:]

    def test_insertion_preserves_prefix(self):
        sql = 'SELECT a, b FROM t WHERE x = 1'
        result = rewrite_pagination(sql, limit=10)
        assert result == sql + ' LIMIT 10'

    def test_semicolon_preserved(self):
        sql = 'SELECT * FROM t;'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM t LIMIT 10;'

    def test_trailing_whitespace_preserved(self):
        sql = 'SELECT * FROM t  '
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM t LIMIT 10  '

    def test_existing_whitespace_before_limit_preserved(self):
        sql = 'SELECT * FROM t  LIMIT 10'
        result = rewrite_pagination(sql, limit=20)
        assert result == 'SELECT * FROM t  LIMIT 20'


class TestUnsupportedStatements:
    """UNION, multi-statement and non-SELECT are rejected."""

    def test_union_rejected(self):
        sql = 'SELECT * FROM a UNION SELECT * FROM b'
        with pytest.raises(UnsupportedStatementError, match='UNION'):
            rewrite_pagination(sql, limit=10)

    def test_union_all_rejected(self):
        sql = 'SELECT * FROM a UNION ALL SELECT * FROM b'
        with pytest.raises(UnsupportedStatementError, match='UNION'):
            rewrite_pagination(sql, limit=10)

    def test_multi_statement_rejected(self):
        sql = 'SELECT * FROM a; SELECT * FROM b'
        with pytest.raises(UnsupportedStatementError, match='single'):
            rewrite_pagination(sql, limit=10)

    def test_non_select_rejected(self):
        sql = 'INSERT INTO t SELECT * FROM s'
        with pytest.raises(UnsupportedStatementError, match='SELECT'):
            rewrite_pagination(sql, limit=10)

    def test_empty_rejected(self):
        with pytest.raises(UnsupportedStatementError):
            rewrite_pagination('', limit=10)

    def test_whitespace_only_rejected(self):
        with pytest.raises(UnsupportedStatementError):
            rewrite_pagination('   ', limit=10)


class TestUnsupportedDialects:
    """TOP, FETCH and ROWNUM dialects are rejected."""

    def test_top_rejected(self):
        sql = 'SELECT TOP 10 * FROM t'
        with pytest.raises(UnsupportedDialectError, match='TOP'):
            rewrite_pagination(sql, limit=20)

    def test_fetch_rejected(self):
        sql = 'SELECT * FROM t FETCH FIRST 10 ROWS ONLY'
        with pytest.raises(UnsupportedDialectError, match='FETCH'):
            rewrite_pagination(sql, limit=20)

    def test_rownum_rejected(self):
        sql = 'SELECT * FROM t WHERE rownum <= 10'
        with pytest.raises(UnsupportedDialectError, match='ROWNUM'):
            rewrite_pagination(sql, limit=20)

    def test_rownum_in_subquery_not_rejected(self):
        sql = 'SELECT * FROM (SELECT rownum r FROM t)'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT * FROM (SELECT rownum r FROM t) LIMIT 10'

    def test_top_without_number_not_rejected(self):
        sql = 'SELECT top FROM t'
        result = rewrite_pagination(sql, limit=10)
        assert result == 'SELECT top FROM t LIMIT 10'


class TestParsePagination:
    """Tests for parse_pagination and the PaginationPlan."""

    def test_parse_limit_and_offset(self):
        plan = parse_pagination('SELECT * FROM t LIMIT 10 OFFSET 5')
        assert plan.limit == PaginationValue('10', False)
        assert plan.offset == PaginationValue('5', False)
        assert plan.span is not None
        assert plan.dialect == 'limit-offset'

    def test_parse_no_pagination(self):
        plan = parse_pagination('SELECT * FROM t')
        assert plan.limit is None
        assert plan.offset is None
        assert plan.span is None

    def test_parse_limit_only(self):
        plan = parse_pagination('SELECT * FROM t LIMIT 10')
        assert plan.limit == PaginationValue('10', False)
        assert plan.offset is None

    def test_parse_offset_only(self):
        plan = parse_pagination('SELECT * FROM t OFFSET 5')
        assert plan.limit is None
        assert plan.offset == PaginationValue('5', False)

    def test_parse_comma_style(self):
        plan = parse_pagination('SELECT * FROM t LIMIT 5, 10')
        assert plan.limit == PaginationValue('10', False)
        assert plan.offset == PaginationValue('5', False)

    def test_parse_limit_all(self):
        plan = parse_pagination('SELECT * FROM t LIMIT ALL')
        assert plan.limit == PaginationValue('ALL', False)

    def test_span_covers_pagination_clause(self):
        sql = 'SELECT * FROM t LIMIT 10 OFFSET 5'
        plan = parse_pagination(sql)
        start, end = plan.span
        assert sql[start:end] == 'LIMIT 10 OFFSET 5'

    def test_span_covers_comma_style(self):
        sql = 'SELECT * FROM t LIMIT 5, 10'
        plan = parse_pagination(sql)
        start, end = plan.span
        assert sql[start:end] == 'LIMIT 5, 10'


class TestRewriteCommaStyle:
    """Rewriting MySQL comma-style LIMIT."""

    def test_rewrite_comma_style(self):
        result = rewrite_pagination('SELECT * FROM t LIMIT 5, 10', limit=20)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET 5'

    def test_rewrite_comma_style_keep_offset(self):
        result = rewrite_pagination(
            'SELECT * FROM t LIMIT 5, 10', limit=20, offset=3)
        assert result == 'SELECT * FROM t LIMIT 20 OFFSET 3'


class TestForUpdate:
    """Locking clauses are handled correctly."""

    def test_insert_before_for_update(self):
        result = rewrite_pagination('SELECT * FROM t FOR UPDATE', limit=10)
        assert result == 'SELECT * FROM t LIMIT 10 FOR UPDATE'

    def test_insert_before_lock_in_share_mode(self):
        result = rewrite_pagination(
            'SELECT * FROM t LOCK IN SHARE MODE', limit=10)
        assert result == 'SELECT * FROM t LIMIT 10 LOCK IN SHARE MODE'

    def test_replace_limit_with_for_update(self):
        result = rewrite_pagination(
            'SELECT * FROM t LIMIT 10 FOR UPDATE', limit=20)
        assert result == 'SELECT * FROM t LIMIT 20 FOR UPDATE'


class TestValueValidation:
    """Invalid values are rejected."""

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError):
            rewrite_pagination('SELECT * FROM t', limit=-1)

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            rewrite_pagination('SELECT * FROM t', offset=-1)

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            rewrite_pagination('SELECT * FROM t', limit=True)

    def test_invalid_placeholder_rejected(self):
        with pytest.raises(ValueError):
            rewrite_pagination('SELECT * FROM t', limit=': bad')


class TestPublicApi:
    """The public API is exported from the sqlparse package."""

    def test_rewrite_pagination_exported(self):
        assert hasattr(sqlparse, 'rewrite_pagination')

    def test_parse_pagination_exported(self):
        assert hasattr(sqlparse, 'parse_pagination')

    def test_exceptions_exported(self):
        assert hasattr(sqlparse, 'PaginationError')
        assert hasattr(sqlparse, 'UnsupportedStatementError')
        assert hasattr(sqlparse, 'UnsupportedDialectError')

    def test_plan_classes_exported(self):
        assert hasattr(sqlparse, 'PaginationPlan')
        assert hasattr(sqlparse, 'PaginationValue')
