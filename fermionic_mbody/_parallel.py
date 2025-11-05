"""
Thin wrapper around multiprocessing.Pool that adds a nice
tqdm progress-bar and lets every other module stay pool-agnostic.
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, Sequence, Tuple, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["chunked"]


# ---------------------------------------------------------------------
def chunked(
    fn: Callable[[T], R],
    iterable: Sequence[T],
    *,
    n_workers: int | None = None,
    description: str | None = None,
) -> list[R]:
    """
    Map fn over iterable in parallel and return the collected list.

    Parameters
    ----------
    fn
        Function executed in each worker.
    iterable
        Sequence of arguments to feed `fn`.
    n_workers
        Number of worker processes (`cpu_count()` by default).
    description
        Optional description for the progress bar.

    Returns
    -------
    list
        The list [fn(x) for x in iterable] with order preserved.
    """
    n_workers = n_workers or cpu_count()

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(fn, iterable),
                total=len(iterable),
                desc=description or fn.__name__,
            )
        )

    return results
