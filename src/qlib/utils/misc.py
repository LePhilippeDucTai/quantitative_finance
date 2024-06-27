"""Utility functions for qlib."""


def to_tuple(t: int | tuple) -> tuple:
    """Convert t to a tuple."""
    if isinstance(t, int):
        return (t,)
    return t
