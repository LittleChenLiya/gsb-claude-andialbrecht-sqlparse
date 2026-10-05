"""Tests for sqlparse.params -- parameter placeholder inventory.

Run locally with pytest (from the repository root)::

    python -m pytest tests/test_params.py -q

Run a single test::

    python -m pytest tests/test_params.py::test_acceptance_string_and_comment_excluded -q

Run with the local worktree on PYTHONPATH if the package is also
installed elsewhere::

    PYTHONPATH=. python -m pytest tests/test_params.py -q
"""

import pytest

import sqlparse
from sqlparse import params
from sqlparse.params import Parameter, ParameterInventory, scan


# ---------------------------------------------------------------------------
# Basic named placeholders
# ---------------------------------------------------------------------------
def test_named_placeholder():
    inv = scan("SELECT :id FROM t")
    assert len(inv) == 1
    p = inv.placeholders[0]
    assert p.name == "id"
    assert p.style == "named"
    assert p.kind == "placeholder"
    assert p.text == ":id"
    assert p.span == (7, 10)
    assert p.statement == 1
    assert p.index == 0
    assert p.order == 0


def test_named_placeholder_span_matches_source():
    sql = "SELECT :id FROM t"
    inv = scan(sql)
    p = inv.placeholders[0]
    assert sql[p.start:p.end] == ":id"


def test_same_named_keeps_multiple_references():
    inv = scan("SELECT :id, :id, :other FROM t")
    assert len(inv.named) == 3
    names = [p.name for p in inv.named]
    assert names == ["id", "id", "other"]
    assert inv.named[0].span != inv.named[1].span
    assert inv.named[0].order == 0
    assert inv.named[1].order == 1


# ---------------------------------------------------------------------------
# Excluded regions
# ---------------------------------------------------------------------------
def test_string_literal_excluded():
    inv = scan("SELECT ':fake' FROM t")
    assert len(inv) == 0


def test_acceptance_string_and_comment_excluded():
    sql = "SELECT :id,':fake' -- :x"
    inv = scan(sql)
    assert len(inv) == 1
    assert inv.placeholders[0].name == "id"
    assert inv.placeholders[0].text == ":id"


def test_line_comment_excluded():
    inv = scan("SELECT :id -- :nope\nFROM t")
    assert len(inv) == 1
    assert inv.placeholders[0].name == "id"


def test_block_comment_excluded():
    inv = scan("SELECT :id /* :nope */ FROM t")
    assert len(inv) == 1
    assert inv.placeholders[0].name == "id"


def test_dollar_quote_excluded():
    sql = "SELECT $tag$:id$tag$ FROM t"
    inv = scan(sql)
    assert len(inv) == 0


def test_dollar_quote_with_named_param_outside():
    sql = "SELECT $tag$:id$tag$; SELECT :name FROM t"
    inv = scan(sql)
    assert len(inv) == 1
    assert inv.placeholders[0].name == "name"
    assert inv.placeholders[0].statement == 2


def test_quoted_identifiers_excluded():
    inv = scan("SELECT `:a`, [:b], \":c\" FROM t")
    assert len(inv) == 0


# ---------------------------------------------------------------------------
# Positional placeholders and statement identity
# ---------------------------------------------------------------------------
def test_positional_placeholder_mysql():
    inv = scan("SELECT ? FROM t", dialect="mysql")
    assert len(inv.positional) == 1
    p = inv.positional[0]
    assert p.name is None
    assert p.style == "positional"
    assert p.kind == "placeholder"
    assert p.text == "?"


def test_two_statements_positional_have_statement_identity():
    sql = "SELECT ? FROM a; SELECT ? FROM b"
    inv = scan(sql, dialect="mysql")
    assert len(inv.positional) == 2
    assert inv.positional[0].statement == 1
    assert inv.positional[1].statement == 2
    assert inv.positional[0].index == 0
    assert inv.positional[1].index == 0
    assert inv.positional[0].order == 0
    assert inv.positional[1].order == 1


def test_dollar_number_postgresql():
    inv = scan("SELECT $1, $2 FROM t", dialect="postgresql")
    assert len(inv.positional) == 2
    assert inv.positional[0].name == "1"
    assert inv.positional[1].name == "2"


def test_dollar_number_sqlite_is_named():
    inv = scan("SELECT $1 FROM t", dialect="sqlite")
    assert len(inv.named) == 1
    assert inv.named[0].name == "1"


# ---------------------------------------------------------------------------
# Question mark dialect distinction
# ---------------------------------------------------------------------------
def test_question_mark_postgresql_is_json_operator():
    inv = scan("SELECT ? FROM t", dialect="postgresql")
    assert len(inv.placeholders) == 0
    assert len(inv.json_operators) == 1
    assert inv.json_operators[0].kind == "json_operator"


def test_question_mark_mysql_is_placeholder():
    inv = scan("SELECT ? FROM t", dialect="mysql")
    assert len(inv.placeholders) == 1
    assert inv.placeholders[0].kind == "placeholder"


def test_question_mark_sqlite_is_placeholder():
    inv = scan("SELECT ? FROM t", dialect="sqlite")
    assert len(inv.placeholders) == 1


def test_question_mark_unknown_dialect_is_unknown():
    for dialect in ("oracle", None):
        inv = scan("SELECT ? FROM t", dialect=dialect)
        assert len(inv.unknowns) == 1
        assert inv.unknowns[0].kind == "unknown"


def test_json_operators_pipe_and_ampersand():
    inv = scan("SELECT ?|, ?&, ? FROM t", dialect="postgresql")
    texts = [op.text for op in inv.json_operators]
    assert texts == ["?|", "?&", "?"]
    assert all(op.kind == "json_operator" for op in inv.json_operators)
    assert inv.json_operators[0].span == (7, 9)
    assert inv.json_operators[1].span == (11, 13)


def test_question_mark_with_space_not_merged():
    inv = scan("SELECT ? | FROM t", dialect="postgresql")
    assert len(inv.json_operators) == 1
    assert inv.json_operators[0].text == "?"


# ---------------------------------------------------------------------------
# @-variables and Python-style placeholders
# ---------------------------------------------------------------------------
def test_at_name_mysql_is_named():
    inv = scan("SELECT @name FROM t", dialect="mysql")
    assert len(inv.named) == 1
    assert inv.named[0].name == "name"


def test_at_name_postgresql_is_unknown():
    inv = scan("SELECT @name FROM t", dialect="postgresql")
    assert len(inv.unknowns) == 1
    assert inv.unknowns[0].kind == "unknown"


def test_python_style_placeholders():
    inv = scan("SELECT %s, %(n)s FROM t", dialect="postgresql")
    assert len(inv.positional) == 1
    assert inv.positional[0].text == "%s"
    assert len(inv.named) == 1
    assert inv.named[0].name == "n"
    assert inv.named[0].text == "%(n)s"


# ---------------------------------------------------------------------------
# Unknown dialect structures
# ---------------------------------------------------------------------------
def test_dollar_number_mysql_is_unknown():
    inv = scan("SELECT $1 FROM t", dialect="mysql")
    assert len(inv.unknowns) == 1
    assert inv.unknowns[0].kind == "unknown"


def test_unknown_structure_marked_not_placeholder():
    inv = scan("SELECT $1 FROM t", dialect="mysql")
    assert len(inv.placeholders) == 0
    assert len(inv.unknowns) == 1


# ---------------------------------------------------------------------------
# Inventory views and metadata
# ---------------------------------------------------------------------------
def test_inventory_views():
    inv = scan(
        "SELECT :id, ? FROM a; SELECT :name FROM b",
        dialect="mysql",
    )
    assert inv.statement_count == 2
    assert len(inv.placeholders) == 3
    assert len(inv.named) == 2
    assert len(inv.positional) == 1
    assert len(inv) == 3
    assert list(inv) == list(inv.parameters)


def test_inventory_repr():
    inv = scan("SELECT :id FROM t")
    r = repr(inv)
    assert "ParameterInventory" in r
    assert "placeholders=1" in r


def test_parameter_to_dict():
    inv = scan("SELECT :id FROM t")
    d = inv.placeholders[0].to_dict()
    assert d["name"] == "id"
    assert d["style"] == "named"
    assert d["kind"] == "placeholder"
    assert d["text"] == ":id"
    assert d["start"] == 7
    assert d["end"] == 10
    assert d["statement"] == 1
    assert d["index"] == 0
    assert d["order"] == 0
    assert d["dialect"] == "unknown"


def test_invalid_dialect_raises():
    with pytest.raises(ValueError):
        scan("SELECT :id FROM t", dialect="bogus")


def test_scan_convenience_function():
    inv = params.scan("SELECT :id FROM t")
    assert isinstance(inv, ParameterInventory)
    assert len(inv.placeholders) == 1


def test_sqlparse_scan_parameters_top_level():
    inv = sqlparse.scan_parameters("SELECT :id FROM t")
    assert isinstance(inv, ParameterInventory)
    assert len(inv.placeholders) == 1


# ---------------------------------------------------------------------------
# Ordering and spans across statements
# ---------------------------------------------------------------------------
def test_global_order_across_statements():
    sql = "SELECT :a FROM t; SELECT :b FROM t; SELECT :c FROM t"
    inv = scan(sql)
    orders = [p.order for p in inv.placeholders]
    assert orders == [0, 1, 2]
    stmts = [p.statement for p in inv.placeholders]
    assert stmts == [1, 2, 3]


def test_index_resets_per_statement():
    sql = "SELECT :a, :b FROM t; SELECT :c FROM t"
    inv = scan(sql)
    assert inv.placeholders[0].index == 0
    assert inv.placeholders[1].index == 1
    assert inv.placeholders[2].index == 0


def test_spans_are_globally_unique_and_ordered():
    sql = "SELECT :a, :b FROM t"
    inv = scan(sql)
    spans = [p.span for p in inv.placeholders]
    assert spans[0][0] < spans[0][1] <= spans[1][0] < spans[1][1]


# ---------------------------------------------------------------------------
# No SQL execution
# ---------------------------------------------------------------------------
def test_no_execution():
    inv = scan("DROP TABLE t; SELECT :id FROM t")
    assert len(inv.placeholders) == 1
    assert inv.placeholders[0].name == "id"
