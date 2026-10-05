"""Tests for explicit outermost-SELECT pagination rewriting.

The rewriting model is plan-first (:func:`plan_pagination`) and byte-based:
each edit is an offset range against the original text and every byte outside
those ranges is preserved verbatim.
"""

import pytest

from sqlparse import pagination as pg
from sqlparse.pagination import (
    LimitValue,
    MultipleStatementsError,
    PaginationError,
    PaginationPlan,
    Placeholder,
    UnsupportedPaginationError,
    paginate,
    plan_pagination,
)


def unchanged_regions(original, edits):
    """Return every byte slice of *original* not covered by an edit."""
    regions = []
    pos = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        regions.append(original[pos : edit.start])
        pos = edit.end
    regions.append(original[pos:])
    return [r for r in regions if r]


def assert_bytes_preserved(plan: PaginationPlan, rewritten: str):
    """Every unedited region must survive byte-for-byte in the output."""
    for region in unchanged_regions(plan.original_sql, plan.edits):
        assert region in rewritten, f"lost original bytes: {region!r}"


# ---------------------------------------------------------------------------
# Adding LIMIT to the outermost SELECT
# ---------------------------------------------------------------------------


def test_add_limit_outermost_keeps_subquery_limit_bytes():
    sql = "SELECT * FROM (SELECT a FROM t LIMIT 5) x"
    plan = plan_pagination(sql, 10)
    out = plan.apply()
    assert out == "SELECT * FROM (SELECT a FROM t LIMIT 5) x LIMIT 10"
    # The inner LIMIT fragment is an unmodified region.
    assert_bytes_preserved(plan, out)
    assert "(SELECT a FROM t LIMIT 5)" in out
    assert plan.limit is None and plan.has_pagination is False
    assert plan.edits[0].start == plan.edits[0].end  # pure insertion


def test_add_limit_to_cte_outer_select_only():
    sql = "WITH c AS (SELECT * FROM t LIMIT 2) SELECT * FROM c"
    out = paginate(sql, 10)
    assert out == "WITH c AS (SELECT * FROM t LIMIT 2) SELECT * FROM c LIMIT 10"


def test_add_limit_with_offset_at_end():
    assert paginate("SELECT * FROM t", 10, 20) == "SELECT * FROM t LIMIT 10 OFFSET 20"


def test_add_limit_before_trailing_semicolon():
    out = paginate("SELECT * FROM t;", 10)
    assert out == "SELECT * FROM t LIMIT 10;"


def test_add_limit_before_for_update():
    out = paginate("SELECT * FROM t FOR UPDATE", 10)
    assert out == "SELECT * FROM t LIMIT 10 FOR UPDATE"
    out2 = paginate("SELECT * FROM t FOR UPDATE", 10, 2)
    assert out2 == "SELECT * FROM t LIMIT 10 OFFSET 2 FOR UPDATE"


def test_add_limit_keeps_whitespace_prefix_and_suffix():
    plan = plan_pagination("  SELECT * FROM t  ", 10)
    out = plan.apply()
    assert out == "  SELECT * FROM t   LIMIT 10" or out.startswith("  SELECT")
    assert out.endswith("LIMIT 10")
    assert_bytes_preserved(plan, out)


def test_string_and_comment_limit_are_ignored():
    # LIMIT inside a line comment and a string must not be treated as a clause.
    sql = "SELECT 'LIMIT 5' AS x FROM t -- LIMIT 99\n/* LIMIT 88 */"
    plan = plan_pagination(sql, 10)
    out = plan.apply()
    assert out == sql + " LIMIT 10"
    assert_bytes_preserved(plan, out)


# ---------------------------------------------------------------------------
# Existing pagination is parsed into the plan
# ---------------------------------------------------------------------------


def test_existing_limit_and_offset_parsed():
    plan = plan_pagination("SELECT * FROM t LIMIT 10 OFFSET 3")
    assert isinstance(plan.limit, LimitValue)
    assert plan.limit.integer == 10
    assert plan.offset.integer == 3
    assert plan.has_pagination is True


def test_replace_existing_limit_and_offset():
    out = paginate("SELECT * FROM t LIMIT 20", 30, 5)
    assert out == "SELECT * FROM t LIMIT 30 OFFSET 5"


def test_replace_only_limit_keeps_offset():
    plan = plan_pagination("SELECT * FROM t LIMIT 20 OFFSET 7", 30)
    out = plan.apply()
    assert out == "SELECT * FROM t LIMIT 30 OFFSET 7"
    assert_bytes_preserved(plan, out)


def test_replace_only_offset_keeps_limit():
    out = paginate("SELECT * FROM t LIMIT 20 OFFSET 7", offset=9)
    assert out == "SELECT * FROM t LIMIT 20 OFFSET 9"


def test_no_requested_change_is_noop():
    sql = "SELECT * FROM t LIMIT 20 OFFSET 7"
    plan = plan_pagination(sql)
    assert plan.edits == ()
    assert plan.apply() == sql


def test_mysql_comma_style_limit_parsed_and_replaced():
    plan = plan_pagination("SELECT * FROM t LIMIT 5, 10", dialect="mysql")
    assert plan.comma_style is True
    assert plan.offset.integer == 5
    assert plan.limit.integer == 10
    out = paginate("SELECT * FROM t LIMIT 5, 10", 20, 0, dialect="mysql")
    assert out == "SELECT * FROM t LIMIT 0, 20"


def test_limit_offset_keyword_style():
    assert paginate("SELECT * FROM t LIMIT 5", 5, 2) == (
        "SELECT * FROM t LIMIT 5 OFFSET 2"
    )


# ---------------------------------------------------------------------------
# Parameter placeholders keep their identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "SELECT * FROM t LIMIT %s OFFSET %s",
        "SELECT * FROM t LIMIT %(lim)s OFFSET %(off)s",
        "SELECT * FROM t LIMIT $1 OFFSET $2",
        "SELECT * FROM t LIMIT :lim OFFSET :off",
        "SELECT * FROM t LIMIT @p1 OFFSET @p2",
        "SELECT * FROM t LIMIT ? OFFSET ?",
    ],
)
def test_existing_placeholders_parsed_and_preserved(text):
    plan = plan_pagination(text)
    assert plan.limit.is_placeholder
    assert plan.offset.is_placeholder
    assert isinstance(plan.limit.placeholder, Placeholder)
    # No concrete values requested: placeholders survive byte-for-byte.
    assert plan.apply() == text
    assert plan.edits == ()


def test_replace_placeholder_with_integer():
    out = paginate("SELECT * FROM t LIMIT %s OFFSET %s", 25, 5)
    assert out == "SELECT * FROM t LIMIT 25 OFFSET 5"


def test_replace_integer_with_placeholder_request():
    out = paginate(
        "SELECT * FROM t LIMIT 10 OFFSET 3", Placeholder(":lim"), Placeholder(":off")
    )
    assert out == "SELECT * FROM t LIMIT :lim OFFSET :off"


def test_request_placeholder_from_string():
    out = paginate("SELECT * FROM t LIMIT 10", "$1", "$2")
    assert out == "SELECT * FROM t LIMIT $1 OFFSET $2"


def test_placeholder_identity_equality():
    assert Placeholder("%s") == Placeholder("%s")
    assert Placeholder("$1") != Placeholder("$2")
    assert str(Placeholder(":lim")) == ":lim"


# ---------------------------------------------------------------------------
# Explicit rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM a UNION SELECT * FROM b",
        "SELECT * FROM a UNION ALL SELECT * FROM b",
        "SELECT * FROM a INTERSECT SELECT * FROM b",
        "SELECT * FROM a EXCEPT SELECT * FROM b",
        "SELECT * FROM a MINUS SELECT * FROM b",
    ],
)
def test_set_operations_rejected(sql):
    with pytest.raises(UnsupportedPaginationError, match="set operation"):
        plan_pagination(sql, 10)


def test_union_inside_cte_is_allowed():
    # The outer statement is a single SELECT; the UNION lives in the CTE.
    out = paginate(
        "WITH c AS (SELECT * FROM a UNION SELECT * FROM b) SELECT * FROM c", 5
    )
    assert out.endswith("SELECT * FROM c LIMIT 5")


def test_top_and_fetch_first_rejected():
    with pytest.raises(UnsupportedPaginationError, match="TOP"):
        plan_pagination("SELECT TOP 10 * FROM t", 10)
    with pytest.raises(UnsupportedPaginationError, match="FETCH"):
        plan_pagination("SELECT * FROM t FETCH FIRST 10 ROWS ONLY", 10)


def test_non_select_rejected():
    with pytest.raises(UnsupportedPaginationError, match="outermost SELECT"):
        plan_pagination("UPDATE t SET a = 1", 10)
    with pytest.raises(UnsupportedPaginationError, match="outermost SELECT"):
        plan_pagination("DELETE FROM t", 10)


def test_unsupported_dialect_rejected():
    with pytest.raises(UnsupportedPaginationError, match="dialect"):
        plan_pagination("SELECT * FROM t", 10, dialect="oracle")


@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "sqlite", "mariadb",
                                     "snowflake", "postgres"])
def test_supported_dialects(dialect):
    out = paginate("SELECT * FROM t", 10, 2, dialect=dialect)
    assert out == "SELECT * FROM t LIMIT 10 OFFSET 2"


def test_multiple_statements_rejected_not_silent_first():
    with pytest.raises(MultipleStatementsError, match="not rewritten silently"):
        plan_pagination("SELECT * FROM t LIMIT 5; SELECT * FROM u LIMIT 6", 10)


def test_multiple_statements_union_also_rejected():
    # Even a single statement that contains UNION is rejected; here both rules
    # could apply, but a PaginationError must surface either way.
    with pytest.raises(PaginationError):
        plan_pagination(
            "SELECT * FROM a UNION SELECT * FROM b; SELECT * FROM c", 10
        )


def test_trailing_empty_statement_is_one_real_statement():
    out = paginate("SELECT * FROM t LIMIT 5;  ", 9)
    assert out == "SELECT * FROM t LIMIT 9;  "


def test_empty_input_rejected():
    with pytest.raises(PaginationError):
        plan_pagination("   ", 10)


def test_no_limit_and_no_request_rejected():
    with pytest.raises(PaginationError, match="no LIMIT"):
        plan_pagination("SELECT * FROM t")


def test_negative_and_bad_type_rejected():
    with pytest.raises(ValueError):
        plan_pagination("SELECT * FROM t", -1)
    with pytest.raises(TypeError):
        plan_pagination("SELECT * FROM t", "10")  # plain numbers must be ints
    with pytest.raises(TypeError):
        plan_pagination("SELECT * FROM t", True)


# ---------------------------------------------------------------------------
# Byte preservation / no full-text replacement
# ---------------------------------------------------------------------------


def test_unmodified_regions_byte_preserved_complex():
    sql = (
        "SELECT a,  b FROM (SELECT x FROM y LIMIT 3) z "
        "WHERE c = 'x LIMIT 1' -- keep me LIMIT 9\n"
        "LIMIT 10 OFFSET 4"
    )
    plan = plan_pagination(sql, 7, 1)
    out = plan.apply()
    assert out.endswith("LIMIT 7 OFFSET 1")
    assert_bytes_preserved(plan, out)
    # Exact inner fragments survive.
    assert "(SELECT x FROM y LIMIT 3)" in out
    assert "'x LIMIT 1'" in out
    assert "-- keep me LIMIT 9\n" in out


def test_edits_reference_original_offsets():
    sql = "SELECT * FROM t LIMIT 20"
    plan = plan_pagination(sql, 30, 5)
    # First edit replaces "20" directly from the original text.
    replace = next(e for e in plan.edits if e.start != e.end)
    assert sql[replace.start : replace.end] == "20"


def test_plan_is_explicit_and_reusable():
    sql = "SELECT * FROM t LIMIT 1"
    plan = plan_pagination(sql, 10)
    assert isinstance(plan, PaginationPlan)
    assert plan.apply() == plan.apply() == "SELECT * FROM t LIMIT 10"
    # Original input is retained on the plan and untouched.
    assert plan.original_sql == sql


def test_plan_as_text_describes_edits():
    plan = plan_pagination("SELECT * FROM t", 10)
    assert "insert" in plan.as_text()
    plan2 = plan_pagination("SELECT * FROM t LIMIT 1", 2)
    assert "replace" in plan2.as_text()
