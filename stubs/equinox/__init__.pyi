# pylint: skip-file

import enum
from io import BufferedReader, BufferedWriter
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Generic,
    Hashable,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    TypeVarTuple,
    overload,
    type_check_only,
)

import nn as nn
from jax import AuxDim
from jax.numpy import ndarray
from jax.tree_util import Leaves

from ._module import *

TCallable = TypeVar("TCallable", bound=Callable[..., Any])
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")
I = TypeVar("I")
Rest = TypeVarTuple("Rest")
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
Shape3 = TypeVarTuple("Shape3")
DType = TypeVar("DType")
DType2 = TypeVar("DType2")
DType3 = TypeVar("DType3")
Dim1 = TypeVar("Dim1")
NumDevicesT = TypeVar("NumDevicesT", bound=int)
Float = TypeVar("Float", bound=float)

@type_check_only
class Grads(Generic[A]): ...

@overload
def filter_jit(*, donate: Literal["all", "warn", "none"]) -> Callable[[TCallable], TCallable]: ...
@overload
def filter_jit(fun: TCallable) -> TCallable: ...
def filter_grad(f: Callable[[A, *Rest], float | ndarray[float]]) -> Callable[[A, *Rest], Grads[A]]: ...
@overload
def filter_value_and_grad(
    has_aux: Literal[False] = False,
) -> Callable[[Callable[[A, *Rest], ndarray[Float]]], Callable[[A, *Rest], tuple[ndarray[Float], Grads[A]]]]: ...
@overload
def filter_value_and_grad(
    has_aux: Literal[True],
) -> Callable[
    [Callable[[A, *Rest], tuple[ndarray[Float], B]]],
    Callable[[A, *Rest], tuple[tuple[ndarray[Float], B], Grads[A]]],
]: ...
@type_check_only
class PMapDecoratorTransform(Protocol):
    @overload
    def __call__(
        self, fun: Callable[[Leaves[A], Leaves[B], C, D], tuple[Leaves[E], Leaves[F], G, H]]
    ) -> Callable[
        [
            Leaves[AuxDim[Any, A]],
            Leaves[AuxDim[Any, B]],
            AuxDim[Any, C],
            AuxDim[Any, D],
        ],
        tuple[
            Leaves[AuxDim[Any, E]],
            Leaves[AuxDim[Any, F]],
            AuxDim[Any, G],
            AuxDim[Any, H],
        ],
    ]: ...
    @overload
    def __call__(
        self, fun: Callable[[A, B, C, D], tuple[E, F, G, H]]
    ) -> Callable[
        [AuxDim[Any, A], AuxDim[Any, B], AuxDim[Any, C], AuxDim[Any, D]],
        tuple[
            AuxDim[Any, E],
            AuxDim[Any, F],
            AuxDim[Any, G],
            AuxDim[Any, H],
        ],
    ]: ...
    @overload
    def __call__(
        self, fun: Callable[[A, B, C, D], tuple[E, F, G, H, I]]
    ) -> Callable[
        [AuxDim[Any, A], AuxDim[Any, B], AuxDim[Any, C], AuxDim[Any, D]],
        tuple[
            AuxDim[Any, E],
            AuxDim[Any, F],
            AuxDim[Any, G],
            AuxDim[Any, H],
            AuxDim[Any, I],
        ],
    ]: ...
    @overload
    def __call__(self, fun: Callable[[A, B], C]) -> Callable[[AuxDim[Any, A], AuxDim[Any, B]], AuxDim[Any, C]]: ...

# Decorator form
@overload
def filter_pmap(
    axis_name: Hashable | None = None,
    donate: Literal["arrays", "warn", "none"] = "none",
) -> PMapDecoratorTransform: ...
@overload
def filter_pmap(
    fun: Callable[[A, B, C, D], tuple[E, F, G, H, I]],
    axis_name: Hashable | None = None,
    donate: Literal["arrays", "warn", "none"] = "none",
) -> Callable[
    [AuxDim[Any, A], AuxDim[Any, B], AuxDim[Any, C], AuxDim[Any, D]],
    tuple[
        AuxDim[Any, E],
        AuxDim[Any, F],
        AuxDim[Any, G],
        AuxDim[Any, H],
        AuxDim[Any, I],
    ],
]: ...
@overload
def filter_pmap(
    fun: Callable[[A, B, C, D], tuple[E, F, G, H]],
    axis_name: Hashable | None = None,
    donate: Literal["arrays", "warn", "none"] = "none",
) -> Callable[
    [AuxDim[Any, A], AuxDim[Any, B], AuxDim[Any, C], AuxDim[Any, D]],
    tuple[
        AuxDim[Any, E],
        AuxDim[Any, F],
        AuxDim[Any, G],
        AuxDim[Any, H],
    ],
]: ...
@overload
def filter_pmap(
    fun: Callable[[Leaves[A], Leaves[B], C, D], tuple[Leaves[E], Leaves[F], G, H]],
    axis_name: Hashable | None = None,
    donate: Literal["arrays", "warn", "none"] = "none",
) -> Callable[
    [
        Leaves[AuxDim[Any, A]],
        Leaves[AuxDim[Any, B]],
        AuxDim[Any, C],
        AuxDim[Any, D],
    ],
    tuple[
        Leaves[AuxDim[Any, E]],
        Leaves[AuxDim[Any, F]],
        AuxDim[Any, G],
        AuxDim[Any, H],
    ],
]: ...
@overload
def filter_pmap(
    fun: Callable[[A, B], C],
    axis_name: Hashable | None = None,
    donate: Literal["arrays", "warn", "none"] = "none",
) -> Callable[[AuxDim[Any, A], AuxDim[Any, B]], AuxDim[Any, C]]: ...
@overload
def filter_vmap(
    fun: Callable[[ndarray[*Shape, A]], B]
) -> Callable[[ndarray[Dim1, *Shape, A]], AuxDim[Dim1, B]]: ...
@overload
def filter_vmap(
    fun: Callable[[ndarray[*Shape, A], ndarray[*Shape2, B]], C]
) -> Callable[[ndarray[Dim1, *Shape, A], ndarray[Dim1, *Shape2, B]], AuxDim[Dim1, C]]: ...
@type_check_only
class PartOf(Generic[A]): ...

M = TypeVar("M", bound=Module)
P = ParamSpec("P")

def filter_checkpoint(
    fun: Callable[P, A], *, prevent_cs: bool = True, policy: Callable[..., bool] | None = None
) -> Callable[P, A]: ...
def partition(pytree: A, filter_spec: Callable[[Any], bool]) -> tuple[PartOf[A], PartOf[A]]: ...
def filter(
    pytree: A,
    filter_spec: Callable[[Any], bool],
    is_leaf: Callable[[Any], bool] | None = None,
    inverse: bool = False,
) -> PartOf[A]: ...
def is_array(element: Any) -> bool: ...
def combine(*pytrees: PartOf[A]) -> A: ...
@overload
def apply_updates(model: M, updates: Grads[M]) -> M: ...
@overload
def apply_updates(model: Grads[M], updates: Grads[M]) -> Grads[M]: ...
def field(static: bool = False) -> Any: ...
def tree_deserialise_leaves(
    path_or_file: str | BufferedReader, pytree: A, filter_spec: Callable[[BinaryIO, A], A] | None = None
) -> A: ...
def tree_serialise_leaves(path_or_file: str | Path | BufferedWriter, pytree: A) -> None: ...
@overload
def tree_at(where: Callable[[A], B], pytree: A, replace: B) -> A: ...
@overload
def tree_at(where: Callable[[A], B], pytree: A, replace_fn: Callable[[B], B]) -> A: ...
def default_deserialise_filter_spec(f: BinaryIO, x: A) -> A: ...
def error_if(x: A, pred: ndarray[bool], msg: str) -> A: ...

class Enumeration(enum.Enum): ...
