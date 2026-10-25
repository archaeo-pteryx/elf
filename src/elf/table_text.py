"""The reader of the plain-text tables ``tables/ept/*.dat`` and ``tables/lpt/*.dat`` (generated offline by tools/pt_autogen).

A table is its '#' header lines (documentation, among them the '# columns:' line) and then one record per line,
whitespace-separated.  The tables are byte-checked by ``python -m pt_autogen check``, so the readers do not re-verify
their structure.  Shared by the table modules ``pt_coeff``, ``pt_matrix`` and ``lpt_tables``.
"""
from fractions import Fraction


def records(path, key=None):
    """The whitespace-split records of the table at ``path`` in file order, the '#' lines skipped.

    With ``key`` (a function of a record, what the line defines) a repeated key raises ValueError naming the file and
    the two lines, for the tables whose duplicates would otherwise be summed or overwritten silently."""
    out, first = [], {}
    with open(path) as fh:
        for number, line in enumerate(fh, 1):
            if line.strip() and not line.startswith('#'):
                record = line.split()
                if key is not None:
                    k = key(record)
                    if k in first:
                        raise ValueError(f'{path} line {number} repeats the key {k} of line {first[k]}')
                    first[k] = number
                out.append(record)
    return out


def fraction(text):
    """The exact rational of a table coefficient ('a/b', 'a' or a decimal)."""
    return Fraction(text)
