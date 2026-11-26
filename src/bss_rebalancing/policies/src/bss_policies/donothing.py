"""The reference without operators: trucks stay at the depot (issue #150)."""

from __future__ import annotations

from bss_policies.base import Controller


class DoNothing(Controller):
    name = "do_nothing"
