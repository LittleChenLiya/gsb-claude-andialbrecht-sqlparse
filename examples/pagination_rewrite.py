"""Explicit pagination rewrite examples (before / after).

Run:
    python examples/pagination_rewrite.py

Only the outermost SELECT is touched; subqueries, CTE bodies, comments and
string contents keep their original bytes.  See ``sqlparse.pagination``.
"""

from sqlparse.pagination import Placeholder, plan_pagination


def show(title, sql, limit=None, offset=None, dialect="postgresql"):
    plan = plan_pagination(sql, limit, offset, dialect=dialect)
    print(f"== {title} ==")
    print(f"before: {sql}")
    print(f"plan  : {plan.as_text()}")
    print(f"after : {plan.apply()}")
    print()


# 1. Add LIMIT 10 to the outer query; the subquery LIMIT 5 is untouched.
show(
    "outer LIMIT added, inner LIMIT preserved byte-for-byte",
    "SELECT * FROM (SELECT a FROM t LIMIT 5) x",
    10,
)

# 2. Existing pagination is parsed first and then replaced.
show(
    "replace existing LIMIT/OFFSET",
    "SELECT * FROM t LIMIT 20",
    30,
    5,
)

# 3. LIMIT inside comments is not part of the plan.
show(
    "comments ignored",
    "SELECT * FROM t -- LIMIT 99\n/* LIMIT 88 */",
    10,
)

# 4. Parameter placeholders keep their identity unless overridden.
show(
    "placeholders preserved",
    "SELECT * FROM t LIMIT %s OFFSET %s",
)
show(
    "placeholder replaced by integers",
    "SELECT * FROM t LIMIT %s OFFSET %s",
    25,
    5,
)
show(
    "request a named placeholder",
    "SELECT * FROM t LIMIT 10",
    Placeholder(":lim"),
)

# 5. MySQL comma-style LIMIT off, lim is parsed and rewritten.
show(
    "MySQL comma-style",
    "SELECT * FROM t LIMIT 5, 10",
    20,
    0,
    dialect="mysql",
)

# 6. A new clause is inserted before FOR UPDATE / a trailing semicolon.
show(
    "insert before FOR UPDATE",
    "SELECT * FROM t FOR UPDATE",
    10,
)
show(
    "insert before trailing semicolon",
    "SELECT * FROM t;",
    10,
)

# 7. Unsupported inputs raise instead of being guessed at:
#      - UNION / INTERSECT / EXCEPT / MINUS
#      - TOP / FETCH FIRST
#      - multiple statements (the first one is never silently rewritten)
from sqlparse.pagination import (  # noqa: E402
    MultipleStatementsError,
    UnsupportedPaginationError,
)

for bad in (
    "SELECT * FROM a UNION SELECT * FROM b",
    "SELECT * FROM t LIMIT 1; SELECT * FROM u LIMIT 2",
):
    try:
        plan_pagination(bad, 10)
    except (UnsupportedPaginationError, MultipleStatementsError) as exc:
        print(f"rejected: {bad!r}\n  -> {type(exc).__name__}: {exc}\n")
