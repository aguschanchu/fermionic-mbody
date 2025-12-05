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
    # Add initialization support
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[R]:
    n_workers = n_workers or cpu_count()

    # Pass initializer to the Pool
    with Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        results = list(
            tqdm(
                pool.imap(fn, iterable),
                total=len(iterable),
                desc=description or fn.__name__,
            )
        )
    return results