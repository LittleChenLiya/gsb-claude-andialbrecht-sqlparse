#!/usr/bin/env python
"""Example: build a parameter placeholder inventory for SQL text.

This script shows how to use :mod:`sqlparse.params` to scan SQL for
named and positional parameter placeholders without executing the SQL.

Run it with::

    python examples/parameter_inventory.py
"""

from sqlparse import scan_parameters


def show(sql, dialect=None):
    print(f"SQL: {sql!r}  (dialect={dialect})")
    inv = scan_parameters(sql, dialect=dialect)
    if inv.placeholders:
        print("  placeholders:")
        for p in inv.placeholders:
            print(
                f"    {p.text!r:10} name={p.name!r:6} "
                f"span={p.span} stmt={p.statement} idx={p.index}"
            )
    if inv.json_operators:
        print("  JSON operators:")
        for p in inv.json_operators:
            print(f"    {p.text!r:6} span={p.span}")
    if inv.unknowns:
        print("  unknown:")
        for p in inv.unknowns:
            print(f"    {p.text!r:6} span={p.span}")
    print()


def main():
    # Named placeholders; strings and comments are excluded.
    show("SELECT :id, ':fake' -- :x")

    # PostgreSQL dollar-quoted literals are excluded too.
    show("SELECT $tag$:id$tag$; SELECT :name FROM t")

    # Positional placeholders get statement identity per statement.
    show("SELECT ? FROM a; SELECT ? FROM b", dialect="mysql")

    # The question mark is a JSON operator in PostgreSQL ...
    show("SELECT ?|, ?& FROM t", dialect="postgresql")

    # ... but a positional placeholder in MySQL/SQLite ...
    show("SELECT ? FROM t", dialect="mysql")

    # ... and unknown when the dialect is not specified.
    show("SELECT ? FROM t")

    # Same-named placeholders keep a record for every occurrence.
    show("SELECT :id, :id, :other FROM t", dialect="postgresql")


if __name__ == "__main__":
    main()
