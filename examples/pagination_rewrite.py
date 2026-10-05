#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Before/after examples for the explicit pagination rewrite.

Run with::

    python examples/pagination_rewrite.py
"""

import os
import sys

# Make sure we use the local sqlparse checkout, not an installed one.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlparse.pagination import (
    PaginationError,
    parse_pagination,
    rewrite_pagination,
)


def show(before, after):
    print(f'BEFORE: {before}')
    print(f'AFTER:  {after}')
    print()


def main():
    # 1. Add LIMIT to a statement without pagination
    show(
        'SELECT * FROM t',
        rewrite_pagination('SELECT * FROM t', limit=10))

    # 2. Replace an existing LIMIT
    show(
        'SELECT * FROM t LIMIT 10',
        rewrite_pagination('SELECT * FROM t LIMIT 10', limit=20))

    # 3. Add both LIMIT and OFFSET
    show(
        'SELECT * FROM t',
        rewrite_pagination('SELECT * FROM t', limit=10, offset=5))

    # 4. Subquery LIMIT is untouched; outer LIMIT is added
    show(
        'SELECT * FROM (SELECT a FROM t LIMIT 5) x',
        rewrite_pagination(
            'SELECT * FROM (SELECT a FROM t LIMIT 5) x', limit=10))

    # 5. LIMIT in a comment does not participate
    show(
        'SELECT * FROM t -- LIMIT 999',
        rewrite_pagination('SELECT * FROM t -- LIMIT 999', limit=10))

    # 6. LIMIT in a string literal does not participate
    show(
        "SELECT * FROM t WHERE x = 'LIMIT 999'",
        rewrite_pagination(
            "SELECT * FROM t WHERE x = 'LIMIT 999'", limit=10))

    # 7. Parameterized pagination keeps placeholder identity
    show(
        'SELECT * FROM t LIMIT ?',
        rewrite_pagination('SELECT * FROM t LIMIT ?', offset=5))

    # 8. Named placeholder is kept
    show(
        'SELECT * FROM t LIMIT :lim OFFSET :off',
        rewrite_pagination(
            'SELECT * FROM t LIMIT :lim OFFSET :off', limit=20))

    # 9. MySQL comma-style LIMIT is parsed into a plan and rewritten
    show(
        'SELECT * FROM t LIMIT 5, 10',
        rewrite_pagination('SELECT * FROM t LIMIT 5, 10', limit=20))

    # 10. FOR UPDATE: LIMIT is inserted before the locking clause
    show(
        'SELECT * FROM t FOR UPDATE',
        rewrite_pagination('SELECT * FROM t FOR UPDATE', limit=10))

    # 11. Parse the existing pagination into a plan
    plan = parse_pagination('SELECT * FROM t LIMIT 10 OFFSET 5')
    print(f'PLAN:   {plan}')
    print()

    # 12. UNION is rejected
    try:
        rewrite_pagination(
            'SELECT * FROM a UNION SELECT * FROM b', limit=10)
    except PaginationError as e:
        print(f'UNION:  rejected -> {e}')
        print()

    # 13. Multi-statement input is rejected (not silently modified)
    try:
        rewrite_pagination('SELECT * FROM a; SELECT * FROM b', limit=10)
    except PaginationError as e:
        print(f'MULTI:  rejected -> {e}')
        print()

    # 14. TOP dialect is rejected
    try:
        rewrite_pagination('SELECT TOP 10 * FROM t', limit=20)
    except PaginationError as e:
        print(f'TOP:    rejected -> {e}')
        print()

    # 15. FETCH dialect is rejected
    try:
        rewrite_pagination(
            'SELECT * FROM t FETCH FIRST 10 ROWS ONLY', limit=20)
    except PaginationError as e:
        print(f'FETCH:  rejected -> {e}')
        print()

    # 16. ROWNUM dialect is rejected
    try:
        rewrite_pagination(
            'SELECT * FROM t WHERE rownum <= 10', limit=20)
    except PaginationError as e:
        print(f'ROWNUM: rejected -> {e}')
        print()


if __name__ == '__main__':
    main()
