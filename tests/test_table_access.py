"""Tests for sqlparse.table_access (offline table access report)."""


from sqlparse.table_access import analyze, format_report


def physical_reads(report, stmt=0):
    return {a.table for a in report.statements[stmt].physical_reads()}


def physical_writes(report, stmt=0):
    return {a.table for a in report.statements[stmt].physical_writes()}


def test_acceptance_cte_join_physical_reads():
    sql = ('WITH t AS (SELECT * FROM raw) '
           'SELECT x.id FROM t x JOIN target y ON x.id=y.id')
    report = analyze(sql)
    assert physical_reads(report) == {'raw', 'target'}
    assert physical_writes(report) == set()
    assert not report.unknowns
    # the CTE reference itself is a logical source, not a physical table
    cte_refs = [a for a in report.statements[0].reads if a.source == 'cte']
    assert [a.table for a in cte_refs] == ['t']
    # scopes: raw is read inside the CTE body, t/target in the main query
    by_table = {a.table: a for a in report.statements[0].reads}
    assert by_table['raw'].scope == 'S1.cte:t'
    assert by_table['t'].scope == 'S1'
    assert by_table['target'].scope == 'S1'


def test_recursive_cte_self_reference_is_not_physical():
    sql = ('WITH RECURSIVE r AS (SELECT 1 UNION ALL SELECT n+1 FROM r) '
           'SELECT * FROM r')
    report = analyze(sql)
    assert physical_reads(report) == set()
    assert {a.table for a in report.statements[0].reads} == {'r'}
    assert all(a.source == 'cte' for a in report.statements[0].reads)


def test_inner_cte_shadows_outer_table_per_scope():
    sql = ('SELECT * FROM t WHERE id IN '
           '(WITH t AS (SELECT * FROM raw2) SELECT id FROM t)')
    report = analyze(sql)
    reads = report.statements[0].reads
    outer = [a for a in reads if a.scope == 'S1']
    inner = [a for a in reads if a.scope != 'S1']
    # outer scope: t is a real table
    assert [(a.table, a.source) for a in outer] == [('t', 'physical')]
    # inner scope: raw2 physical, t resolves to the shadowing CTE
    assert ('raw2', 'physical') in [(a.table, a.source) for a in inner]
    assert ('t', 'cte') in [(a.table, a.source) for a in inner]
    assert physical_reads(report) == {'t', 'raw2'}


def test_aliases_and_column_qualifiers_are_not_tables():
    sql = ('SELECT a.id, b.name FROM users a JOIN orders b '
           'ON a.id = b.user_id')
    report = analyze(sql)
    assert physical_reads(report) == {'users', 'orders'}
    tables = {a.table for a in report.accesses}
    assert 'a' not in tables and 'b' not in tables
    assert 'a.id' not in tables and 'b.name' not in tables


def test_quoted_and_schema_qualified_segments_preserved():
    sql = 'SELECT * FROM "My Schema"."Order Details" x'
    report = analyze(sql)
    (acc,) = report.statements[0].reads
    assert acc.table == 'My Schema.Order Details'
    assert acc.raw == '"My Schema"."Order Details"'
    assert sql[acc.span[0]:acc.span[1]] == '"My Schema"."Order Details"'


def test_insert_write_and_select_read_are_marked_separately():
    sql = ('INSERT INTO "mart"."Daily Stats" (id) '
           'SELECT o.id FROM staging.orders o '
           'WHERE o.id NOT IN (SELECT id FROM ops.blacklist)')
    report = analyze(sql)
    stmt = report.statements[0]
    assert stmt.type == 'INSERT'
    assert physical_writes(report) == {'mart.Daily Stats'}
    assert physical_reads(report) == {'staging.orders', 'ops.blacklist'}
    write = stmt.writes[0]
    assert write.scope == 'S1'
    nested = next(a for a in stmt.reads if a.table == 'ops.blacklist')
    assert nested.scope == 'S1.q1'


def test_update_with_from_subquery():
    sql = ('UPDATE ops.users u SET flag = 1 '
           'FROM (SELECT user_id FROM raw.logs) s '
           'WHERE s.user_id = u.id')
    report = analyze(sql)
    assert physical_writes(report) == {'ops.users'}
    assert physical_reads(report) == {'raw.logs'}


def test_delete_with_subquery():
    sql = 'DELETE FROM sessions WHERE id IN (SELECT id FROM old)'
    report = analyze(sql)
    assert physical_writes(report) == {'sessions'}
    assert physical_reads(report) == {'old'}


def test_nested_subquery_scopes():
    sql = ('SELECT * FROM a WHERE x IN '
           '(SELECT y FROM b WHERE z IN (SELECT w FROM c))')
    report = analyze(sql)
    by_table = {a.table: a for a in report.statements[0].reads}
    assert by_table['a'].scope == 'S1'
    assert by_table['b'].scope == 'S1.q1'
    assert by_table['c'].scope == 'S1.q1.q1'


def test_spans_point_at_raw_text():
    sql = ('WITH t AS (SELECT * FROM raw) '
           'SELECT x.id FROM t x JOIN target y ON x.id=y.id; '
           'INSERT INTO dst SELECT * FROM t')
    report = analyze(sql)
    assert len(report.statements) == 2
    for acc in report.accesses:
        assert sql[acc.span[0]:acc.span[1]] == acc.raw
    # second statement: t is not a CTE there, it is a physical table
    assert physical_reads(report, 1) == {'t'}
    assert physical_writes(report, 1) == {'dst'}


def test_comma_join_and_join_variants():
    sql = 'SELECT * FROM a, b.c d LEFT JOIN e ON 1=1'
    report = analyze(sql)
    assert physical_reads(report) == {'a', 'b.c', 'e'}


def test_unknown_constructs_are_reported_not_silently_parsed():
    # MERGE: unsupported statement type
    report = analyze('MERGE INTO tgt USING src ON tgt.id = src.id '
                     'WHEN MATCHED THEN UPDATE SET x = 1')
    assert report.unknowns
    assert 'MERGE' in report.unknowns[0].reason
    assert report.accesses == []
    # table-valued function
    report = analyze('SELECT * FROM generate_series(1, 10) g')
    assert len(report.unknowns) == 1
    assert 'function' in report.unknowns[0].reason
    # LATERAL
    report = analyze('SELECT * FROM t LATERAL VIEW explode(x) v AS y')
    assert physical_reads(report) == {'t'}
    assert any('LATERAL' in u.reason for u in report.unknowns)


def test_format_report_attribution_lines():
    sql = ('WITH t AS (SELECT * FROM raw) '
           'SELECT x.id FROM t x JOIN target y ON x.id=y.id')
    text = format_report(analyze(sql))
    assert f'== S1 SELECT span=0:{len(sql)} ==' in text
    assert ('S1.cte:t | read | physical | raw | span=25:28 | raw=raw'
            in text)
    assert 'S1 | read | cte | t |' in text
    assert 'S1 | read | physical | target |' in text


def test_demo_sqls_run():
    from sqlparse.table_access import _DEMO_SQL
    assert len(_DEMO_SQL) == 3
    for sql in _DEMO_SQL:
        report = analyze(sql)
        assert report.statements
    # demo 1: the acceptance statement
    assert physical_reads(analyze(_DEMO_SQL[0])) == {'raw', 'target'}
    # demo 3: MERGE is explicitly unknown
    assert analyze(_DEMO_SQL[2]).unknowns
