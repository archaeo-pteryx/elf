"""The shared fixture of the generator's tests, the LPT chain derived once per session."""
import pytest

from pt_autogen.__main__ import generate


@pytest.fixture(scope='session')
def lpt_derived():
    """The LPT part of ``generate`` (lpt_cumulants, lpt_components, lpt_rows, lpt_angular)."""
    return generate(('lpt',))[0]
