"""Tests for sqlparse.params (placeholder inventory).

Run locally with:

    uv run --group dev --python 3.11 pytest tests/test_params.py

or, without uv:

    python -m pytest tests/test_params.py
"""

import pytest

from sqlparse.params import DIALECTS, extract_params


def test_named_only_string_and_comment_excluded():
    # Acceptance: SELECT :id,':fake' -- :x  collects only :id.
    inv = extract_params("SELECT :id, ':fake' -- :x\n")
    assert [(p.kind, p.name, p.text) for p in inv.params] == [
        ("named", "id", ":id")
    ]
    ref = inv.params[0]
    assert ref.statement == 1
    assert ref.order == 1
    sql = "SELECT :id, ':fake' -- :x\n"
    assert sql[ref.span[0]:ref.span[1]] == ":id"


def test_dollar_quoted_body_excluded():
    sql = "SELECT $$body :fake$$, $tag$also :no$tag$, :real"
    inv = extract_params(sql, dialect="postgresql")
    assert [p.text for p in inv.params] == [":real"]
    assert sql[inv.params[0].span[0]:inv.params[0].span[1]] == ":real"


def test_quoted_identifiers_excluded():
    sql = 'SELECT ":not_a_param", `:no`, [:no] FROM t WHERE a = :yes'
    inv = extract_params(sql)
    assert [p.text for p in inv.params] == [":yes"]


def test_multiline_comment_excluded():
    sql = "SELECT /* :hidden */ :shown"
    inv = extract_params(sql)
    assert [p.text for p in inv.params] == [":shown"]


def test_two_statements_positional_identity():
    sql = "SELECT ?; SELECT ?, ?"
    inv = extract_params(sql)
    assert [(p.statement, p.order) for p in inv.params] == [
        (1, 1), (2, 2), (2, 3),
    ]
    assert all(p.kind == "positional" for p in inv.params)
    # spans point back into the original text
    for p in inv.params:
        assert sql[p.span[0]:p.span[1]] == "?"
    # every question mark is listed separately with its dialect note
    assert len(inv.question_marks) == 3
    assert all("positional placeholder" in q.note
               for q in inv.question_marks)


def test_same_name_multiple_references_preserved():
    sql = "SELECT :id, :id, :other, :id"
    inv = extract_params(sql)
    by_name = inv.by_name()
    assert sorted(by_name) == ["id", "other"]
    assert len(by_name["id"]) == 3
    assert [p.order for p in by_name["id"]] == [1, 2, 4]
    # each reference keeps its own span
    spans = [p.span for p in by_name["id"]]
    assert len(set(spans)) == 3


@pytest.mark.parametrize("dialect", ["generic", "mysql", "sqlite"])
def test_question_mark_placeholder_dialects(dialect):
    inv = extract_params("SELECT ?", dialect=dialect)
    assert inv.params[0].kind == "positional"
    assert inv.question_marks[0].note


def test_question_mark_postgresql_json_operator():
    sql = "SELECT data ? 'key'"
    inv = extract_params(sql, dialect="postgresql")
    # not a placeholder in PostgreSQL: kept as unknown, listed apart
    assert inv.params[0].kind == "unknown"
    assert "JSON" in inv.question_marks[0].note


def test_postgresql_numbered_placeholder():
    inv = extract_params("SELECT $1, $2", dialect="postgresql")
    assert [p.kind for p in inv.params] == ["positional", "positional"]


def test_numbered_placeholder_unknown_in_mysql():
    inv = extract_params("SELECT $1", dialect="mysql")
    assert inv.params[0].kind == "unknown"


def test_unknown_shape_marked_unknown():
    inv = extract_params("SELECT $name")
    assert inv.params[0].kind == "unknown"
    assert "unrecognized" in inv.params[0].note


def test_python_format_placeholders():
    inv = extract_params("SELECT %s, %(item)s")
    assert [(p.kind, p.name) for p in inv.params] == [
        ("positional", None), ("named", "item"),
    ]


def test_cast_and_assignment_are_not_placeholders():
    sql = "SELECT a::int, b := 1"
    inv = extract_params(sql, dialect="postgresql")
    assert inv.params == []


def test_unknown_dialect_rejected():
    with pytest.raises(ValueError):
        extract_params("SELECT 1", dialect="oracle-ish")


def test_dialects_listed():
    assert set(DIALECTS) == {"generic", "mysql", "sqlite", "postgresql"}


def test_render_mentions_question_mark_section():
    inv = extract_params("SELECT ?")
    text = inv.render()
    assert "question marks" in text
    assert "positional placeholder" in text
