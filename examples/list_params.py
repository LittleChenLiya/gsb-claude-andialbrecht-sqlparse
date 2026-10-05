"""List parameter placeholders of a SQL script (never executes SQL).

Run from the repository root:

    python examples/list_params.py

Sample output (dialect=generic)::

    dialect: generic
    stmt  ord  kind        name      span          text
    1     1    named       id        (7, 10)       :id
    1     2    positional  -         (51, 52)      ?   # positional placeholder (generic SQL)
    1     3    named       id        (75, 78)      :id
    2     4    positional  -         (87, 88)      ?   # positional placeholder (generic SQL)
    2     5    positional  -         (90, 91)      ?   # positional placeholder (generic SQL)
    question marks (dialect interpretation):
      stmt 1 span (51, 52): positional placeholder (generic SQL)
      stmt 2 span (87, 88): positional placeholder (generic SQL)
      stmt 2 span (90, 91): positional placeholder (generic SQL)

Only ``:id`` is collected from the first statement: ``':fake'`` is a
string literal, ``-- :x`` is a comment and the dollar-quoted body is a
PostgreSQL literal, so none of them are parameters.  The two ``?`` in
the second statement carry their own statement number and occurrence
order.  Re-run with ``dialect="postgresql"`` to see the ``?`` entries
re-interpreted as the JSON existence operator.
"""

import os
import sys

# Prefer the source checkout over an installed (older) sqlparse,
# since sqlparse.params is not part of a released version yet.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlparse.params import extract_params

SQL = """\
SELECT :id, ':fake' -- :x
, $$body :not_a_param$$, ?
FROM users WHERE id = :id;
SELECT ?, ?;
"""

if __name__ == "__main__":
    for dialect in ("generic", "postgresql"):
        inventory = extract_params(SQL, dialect=dialect)
        print(inventory.render())
        print()
