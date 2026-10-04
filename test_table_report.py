"""table_report 的局部 pytest 验收测试。

运行：pytest test_table_report.py -v
"""

import pytest

from table_report import analyze_sql


def names(refs):
    return sorted({r.qualified for r in refs})


def by_scope(refs):
    out = {}
    for r in refs:
        out.setdefault(r.scope, []).append(r.qualified)
    return out


def test_acceptance_sql():
    sql = "WITH t AS (SELECT * FROM raw) SELECT x.id FROM t x JOIN target y ON x.id=y.id"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "SELECT"
    assert rep.physical_set("read") == ["raw", "target"]
    cte_refs = [r for r in rep.reads if r.source == "cte"]
    assert names(cte_refs) == ["t"]
    assert cte_refs[0].cte_name == "t"
    assert rep.physical_set("write") == []


def test_cte_shadow_same_name_external_table():
    sql = "WITH t AS (SELECT * FROM t) SELECT * FROM t"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["t"]
    scoped = by_scope(rep.reads)
    cte_scope = [s for s in rep.scopes if s["kind"] == "cte"][0]
    assert "t" in scoped[cte_scope["id"]]
    cte_body_ref = [r for r in rep.reads
                    if r.scope == cte_scope["id"] and r.qualified == "t"][0]
    assert cte_body_ref.source == "physical"
    main_ref = [r for r in rep.reads
                if r.scope == "root" and r.qualified == "t"][0]
    assert main_ref.source == "cte"


def test_recursive_cte_self_ref_not_physical():
    sql = ("WITH RECURSIVE c AS "
           "(SELECT n FROM nums UNION ALL SELECT n+1 FROM c WHERE n<10) "
           "SELECT * FROM c")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["nums"]
    cte_scope = [s for s in rep.scopes if s["kind"] == "cte"][0]
    self_ref = [r for r in rep.reads
                if r.scope == cte_scope["id"] and r.qualified == "c"]
    assert len(self_ref) == 1
    assert self_ref[0].source == "cte"
    main_ref = [r for r in rep.reads
                if r.scope == "root" and r.qualified == "c"][0]
    assert main_ref.source == "cte"


def test_insert_select_marked_separately():
    sql = "INSERT INTO dbo.target (id) SELECT id FROM src s WHERE s.id > 0"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "INSERT"
    assert rep.physical_set("write") == ["dbo.target"]
    assert rep.physical_set("read") == ["src"]
    assert names(rep.writes) == ["dbo.target"]
    assert names(rep.reads) == ["src"]


def test_update_from():
    sql = "UPDATE t SET a=1 FROM u WHERE t.id=u.id"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "UPDATE"
    assert rep.physical_set("write") == ["t"]
    assert rep.physical_set("read") == ["u"]


def test_delete():
    sql = "DELETE FROM logs WHERE created < '2020-01-01'"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "DELETE"
    assert rep.physical_set("write") == ["logs"]
    assert rep.physical_set("read") == []


def test_insert_values_no_reads():
    sql = "INSERT INTO t (a, b) VALUES (1, 2)"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("write") == ["t"]
    assert rep.physical_set("read") == []


def test_quoted_and_schema_qualified_segments():
    sql = 'SELECT * FROM "Raw" r JOIN dbo."Order" o ON r.id = o.id'
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["Raw", "dbo.Order"]
    raw = [r for r in rep.reads if r.qualified == "Raw"][0]
    assert raw.segments == ["Raw"]
    assert raw.alias == "r"
    order = [r for r in rep.reads if r.qualified == "dbo.Order"][0]
    assert order.segments == ["dbo", "Order"]
    assert order.alias == "o"


def test_bracket_and_backtick_quoting():
    sql = "SELECT * FROM [dbo].[T] JOIN `sch`.`Tbl` ON 1=1"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["dbo.T", "sch.Tbl"]
    t = [r for r in rep.reads if r.qualified == "dbo.T"][0]
    assert t.segments == ["dbo", "T"]
    tbl = [r for r in rep.reads if r.qualified == "sch.Tbl"][0]
    assert tbl.segments == ["sch", "Tbl"]


def test_nested_subquery_scopes():
    sql = ("SELECT * FROM (SELECT id FROM inner_tab) sub "
           "WHERE sub.id IN (SELECT id FROM keep_tab)")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["inner_tab", "keep_tab"]
    sub_scopes = [s for s in rep.scopes if s["kind"] == "subquery"]
    assert len(sub_scopes) == 2
    by_sc = by_scope(rep.reads)
    assert "inner_tab" in by_sc[sub_scopes[0]["id"]]
    assert "keep_tab" in by_sc[sub_scopes[1]["id"]]


def test_subquery_in_select_list():
    sql = "SELECT (SELECT max FROM t) AS m FROM dual"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["dual", "t"]
    assert len([s for s in rep.scopes if s["kind"] == "subquery"]) == 1


def test_cte_visible_inside_nested_subquery():
    sql = ("WITH t AS (SELECT id FROM base) "
           "SELECT * FROM (SELECT id FROM t) x")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["base"]
    sub = [s for s in rep.scopes if s["kind"] == "subquery"][0]
    sub_ref = [r for r in rep.reads if r.scope == sub["id"]][0]
    assert sub_ref.source == "cte"


def test_column_qualifiers_not_tables():
    sql = "SELECT x.id, y.z FROM t x JOIN u y ON x.id = y.z WHERE x.id > 0"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["t", "u"]


def test_implicit_alias_not_extra_table():
    sql = "SELECT * FROM raw r"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["raw"]
    assert len(rep.reads) == 1
    assert rep.reads[0].alias == "r"


@pytest.mark.parametrize("join_kw", [
    "JOIN", "INNER JOIN", "LEFT JOIN", "LEFT OUTER JOIN",
    "RIGHT JOIN", "FULL JOIN", "CROSS JOIN", "NATURAL JOIN",
])
def test_join_variants(join_kw):
    sql = "SELECT * FROM a %s b ON a.id = b.id" % join_kw
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["a", "b"]


def test_comma_join():
    sql = "SELECT * FROM a, b WHERE a.id = b.id"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["a", "b"]


def test_unsupported_statement_reported():
    sql = "CREATE TABLE t (id INT)"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "UNKNOWN"
    assert rep.physical_set("read") == []
    assert rep.physical_set("write") == []
    assert any(u.kind == "unsupported_statement" for u in rep.unknowns)


def test_table_function_reported_unknown():
    sql = "SELECT * FROM generate_series(1, 10) g"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == []
    assert any(u.kind == "table_function" for u in rep.unknowns)


def test_unknown_statement_does_not_claim_parse():
    sql = ("MERGE INTO t USING s ON t.id = s.id "
           "WHEN MATCHED THEN UPDATE SET t.a = s.a")
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "UNKNOWN"
    assert rep.physical_set("read") == []


def test_text_ranges():
    sql = "SELECT * FROM raw"
    rep = analyze_sql(sql)[0]
    ref = rep.reads[0]
    assert ref.text == "raw"
    assert sql[ref.range[0]:ref.range[1]] == "raw"
    assert rep.range == (0, len(sql))


def test_multiple_statements():
    sql = "SELECT * FROM a; SELECT * FROM b;"
    reports = analyze_sql(sql)
    assert len(reports) == 2
    assert reports[0].physical_set("read") == ["a"]
    assert reports[1].physical_set("read") == ["b"]


def test_demo_runs():
    from table_report import demo
    out = demo()
    assert "raw" in out and "target" in out
    assert "archive" in out
    assert "dbo" in out or "Order" in out


def test_delete_using_reads_source():
    sql = "DELETE FROM t USING u WHERE t.id = u.id"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("write") == ["t"]
    assert rep.physical_set("read") == ["u"]


def test_cte_with_column_list():
    sql = "WITH a(x) AS (SELECT 1), b AS (SELECT * FROM a) SELECT * FROM b"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == []
    cte_refs = [r for r in rep.reads if r.source == "cte"]
    assert names(cte_refs) == ["a", "b"]


def test_multiple_ctes_visibility():
    sql = ("WITH a AS (SELECT id FROM base), "
           "b AS (SELECT * FROM a) "
           "SELECT * FROM b")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["base"]
    # b 的作用域里 a 是逻辑来源
    b_scope = [s for s in rep.scopes if s["kind"] == "cte"][1]
    a_ref = [r for r in rep.reads if r.scope == b_scope["id"]][0]
    assert a_ref.source == "cte"


def test_lateral_subquery():
    sql = ("SELECT * FROM a LEFT JOIN LATERAL "
           "(SELECT * FROM b WHERE b.id = a.id) x ON true")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["a", "b"]
    assert len([s for s in rep.scopes if s["kind"] == "subquery"]) == 1


def test_select_into_write():
    sql = "SELECT * INTO new_t FROM old_t"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("write") == ["new_t"]
    assert rep.physical_set("read") == ["old_t"]


def test_explain_transparent():
    sql = "EXPLAIN SELECT * FROM a JOIN b ON a.id = b.id"
    rep = analyze_sql(sql)[0]
    assert rep.stmt_type == "SELECT"
    assert rep.physical_set("read") == ["a", "b"]


def test_dml_inside_cte():
    sql = ("WITH ins AS (INSERT INTO t (id) VALUES (1) RETURNING id) "
           "SELECT * FROM ins")
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("write") == ["t"]
    assert rep.physical_set("read") == []
    main_ref = [r for r in rep.reads if r.scope == "root"][0]
    assert main_ref.source == "cte"


def test_union_all():
    sql = "SELECT * FROM a UNION ALL SELECT * FROM b"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["a", "b"]


def test_keyword_classified_table_name():
    # archive 在 sqlparse 中被标为 Keyword，但它是表名
    sql = "SELECT * FROM archive a"
    rep = analyze_sql(sql)[0]
    assert rep.physical_set("read") == ["archive"]
    assert rep.reads[0].alias == "a"
