from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

from . import notebook as notebook

A = TypeVar("A")

class tqdm(Generic[A]):
    def __iter__(self) -> Iterator[A]: ...
    def __next__(self) -> A: ...
    def __init__(
        self, iterable: Iterable[A] | None = None, total: int | None = None, unit: str = "it", desc: str = ""
    ) -> None: ...
    def __enter__(self) -> tqdm[A]: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    def set_postfix(self, **kwargs: Any) -> None: ...
    def update(self, n: int = 1) -> None: ...