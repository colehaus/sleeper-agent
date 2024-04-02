# pylint: skip-file

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import (
    Any,
    Generic,
    Literal,
    NewType,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    overload,
    type_check_only,
)

from numpy import ndarray

from . import config as config
from . import debug as debug
from . import dtypes as dtypes
from . import lax as lax
from . import nn as nn
from . import numpy as numpy
from . import random as random
from . import scipy as scipy
from . import tree_util as tree_util
from .core import ClosedJaxpr

CallableT = TypeVar("CallableT", bound=Callable[..., Any])
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
Tuple = TypeVar("Tuple", bound=tuple[Any, ...])
Tuple2 = TypeVar("Tuple2", bound=tuple[Any, ...])
Tuple3 = TypeVar("Tuple3", bound=tuple[Any, ...])
As = TypeVarTuple("As")
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
Shape3 = TypeVarTuple("Shape3")
Shape4 = TypeVarTuple("Shape4")
Shape5 = TypeVarTuple("Shape5")
Shape6 = TypeVarTuple("Shape6")
Shape7 = TypeVarTuple("Shape7")
Dim1 = TypeVar("Dim1")
Dim2 = TypeVar("Dim2")
Dim3 = TypeVar("Dim3")
Dim4 = TypeVar("Dim4")
Dim5 = TypeVar("Dim5")
Dim6 = TypeVar("Dim6")
Dim7 = TypeVar("Dim7")
Dim8 = TypeVar("Dim8")
Dim9 = TypeVar("Dim9")
Dim10 = TypeVar("Dim10")
OutDim = TypeVar("OutDim")
InDim = TypeVar("InDim")
Rest = TypeVarTuple("Rest")
DType = TypeVar("DType")
DType2 = TypeVar("DType2")
DType3 = TypeVar("DType3")
DType4 = TypeVar("DType4")
DType5 = TypeVar("DType5")
DType6 = TypeVar("DType6")
DType7 = TypeVar("DType7")

# We only use this for PRNG keys
Array: TypeAlias = ndarray[Literal[2], int]

# This is not quite right but anything other than `Callable` for the return type
# "fixes" the type variables at creation time
def custom_jvp(fun: Callable[[A], B]) -> Callable[[A], B]: ...
@type_check_only
class AuxDim(Generic[*Shape, A]):
    def __iter__(self: AuxDim[Dim1, A]) -> Iterator[A]: ...

@overload
def vmap(
    fun: Callable[
        [
            ndarray[Dim1, *Shape, DType],
            ndarray[Dim2, *Shape2, DType2],
            ndarray[Dim3, *Shape3, DType3],
            ndarray[*Shape4, DType4] | None,
        ],
        ndarray[Dim4, *Shape5, DType5],
    ],
    in_axes: tuple[Literal[1], Literal[1], Literal[1], Literal[0]],
    out_axes: Literal[1],
) -> Callable[
    [
        ndarray[Dim1, Dim5, *Shape, DType],
        ndarray[Dim2, Dim5, *Shape2, DType2],
        ndarray[Dim3, Dim5, *Shape3, DType3],
        ndarray[Dim5, *Shape4, DType4] | None,
    ],
    ndarray[Dim4, Dim5, *Shape5, DType5],
]: ...
@overload
def vmap(
    fun: Callable[
        [
            ndarray[Dim1, *Shape, DType],
            ndarray[Dim2, *Shape2, DType2],
            ndarray[Dim3, *Shape3, DType3],
            ndarray[*Shape4, DType4],
            ndarray[*Shape5, DType5] | None,
        ],
        ndarray[Dim4, *Shape6, DType6],
    ],
    in_axes: tuple[Literal[1], Literal[1], Literal[1], Literal[0], Literal[0]],
    out_axes: Literal[1],
) -> Callable[
    [
        ndarray[Dim1, Dim5, *Shape, DType],
        ndarray[Dim2, Dim5, *Shape2, DType2],
        ndarray[Dim3, Dim5, *Shape3, DType3],
        ndarray[Dim5, *Shape4, DType4],
        ndarray[Dim5, *Shape5, DType5] | None,
    ],
    ndarray[Dim4, Dim5, *Shape6, DType6],
]: ...
@overload
def vmap(
    fun: Callable[[ndarray[*Shape, DType]], ndarray[*Shape2, DType2]]
) -> Callable[[ndarray[Dim1, *Shape, DType]], ndarray[Dim1, *Shape2, DType2]]: ...
@overload
def vmap(
    fun: Callable[[ndarray[*Shape, DType], ndarray[*Shape2, DType2] | None], ndarray[*Shape3, DType3]]
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2] | None], ndarray[Dim1, *Shape3, DType3]
]: ...
@overload
def vmap(
    fun: Callable[[ndarray[*Shape, DType], ndarray[*Shape2, DType2]], ndarray[*Shape3, DType3]]
) -> Callable[[ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2]], ndarray[Dim1, *Shape3, DType3]]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2], ndarray[*Shape3, DType3]],
        ndarray[*Shape4, DType4],
    ],
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2], ndarray[Dim1, *Shape3, DType3]],
    ndarray[Dim1, *Shape4, DType4],
]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2], ndarray[*Shape3, DType3], ndarray[*Shape4, DType4]],
        ndarray[*Shape5, DType5],
    ],
) -> Callable[
    [
        ndarray[Dim1, *Shape, DType],
        ndarray[Dim1, *Shape2, DType2],
        ndarray[Dim1, *Shape3, DType3],
        ndarray[Dim1, *Shape4, DType4],
    ],
    ndarray[Dim1, *Shape5, DType5],
]: ...
@overload
def vmap(
    fun: Callable[
        [
            ndarray[*Shape, DType],
            ndarray[*Shape2, DType2],
            ndarray[*Shape3, DType3],
            ndarray[*Shape4, DType4],
            ndarray[*Shape5, DType5],
            ndarray[*Shape6, DType6],
        ],
        Tuple,
    ],
) -> Callable[
    [
        ndarray[Dim1, *Shape, DType],
        ndarray[Dim1, *Shape2, DType2],
        ndarray[Dim1, *Shape3, DType3],
        ndarray[Dim1, *Shape4, DType4],
        ndarray[Dim1, *Shape5, DType5],
        ndarray[Dim1, *Shape6, DType6],
    ],
    AuxDim[Dim1, Tuple],
]: ...
@overload
def vmap(
    fun: Callable[
        [
            ndarray[*Shape, DType],
            ndarray[*Shape2, DType2],
            ndarray[*Shape3, DType3],
            ndarray[*Shape4, DType4],
            ndarray[*Shape5, DType5],
            ndarray[*Shape6, DType6],
        ],
        ndarray[*Shape7, DType7],
    ],
) -> Callable[
    [
        ndarray[Dim1, *Shape, DType],
        ndarray[Dim1, *Shape2, DType2],
        ndarray[Dim1, *Shape3, DType3],
        ndarray[Dim1, *Shape4, DType4],
        ndarray[Dim1, *Shape5, DType5],
        ndarray[Dim1, *Shape6, DType6],
    ],
    ndarray[Dim1, *Shape7, DType7],
]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2], ndarray[*Shape3, DType3], ndarray[*Shape4, DType4]],
        Tuple,
    ],
) -> Callable[
    [
        ndarray[Dim1, *Shape, DType],
        ndarray[Dim1, *Shape2, DType2],
        ndarray[Dim1, *Shape3, DType3],
        ndarray[Dim1, *Shape4, DType4],
    ],
    AuxDim[Dim1, Tuple],
]: ...
@overload
def vmap(
    fun: Callable[[ndarray[*Shape, DType]], Tuple]
) -> Callable[[ndarray[Dim1, *Shape, DType]], AuxDim[Dim1, Tuple]]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2]],
        tuple[ndarray[*Shape3, DType3], ndarray[*Shape4, DType4]],
    ],
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2]],
    tuple[ndarray[Dim1, *Shape3, DType3], ndarray[Dim1, *Shape4, DType4]],
]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2]],
        tuple[ndarray[*Shape3, DType3], ndarray[*Shape4, DType4], ndarray[*Shape5, DType5]],
    ],
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2]],
    tuple[ndarray[Dim1, *Shape3, DType3], ndarray[Dim1, *Shape4, DType4], ndarray[Dim1, *Shape5, DType5]],
]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2], ndarray[*Shape3, DType3]],
        tuple[ndarray[*Shape4, DType4], tuple[ndarray[*Shape5, DType5], ndarray[*Shape6, DType6]]],
    ],
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2], ndarray[Dim1, *Shape3, DType3]],
    tuple[ndarray[Dim1, *Shape4, DType4], tuple[ndarray[Dim1, *Shape5, DType5], ndarray[Dim1, *Shape6, DType6]]],
]: ...
@overload
def vmap(
    fun: Callable[[Tuple, ndarray[*Shape2, DType2]], ndarray[*Shape, DType]]
) -> Callable[[AuxDim[Dim1, Tuple], ndarray[Dim1, *Shape2, DType2]], ndarray[Dim1, *Shape, DType]]: ...
def jit(func: CallableT) -> CallableT: ...
def jacrev(
    func: Callable[[ndarray[InDim, float]], ndarray[OutDim, float]]
) -> Callable[[ndarray[InDim, float]], ndarray[InDim, OutDim, float]]: ...
def jacfwd(
    func: Callable[[ndarray[InDim, float]], ndarray[OutDim, float]]
) -> Callable[[ndarray[InDim, float]], ndarray[InDim, OutDim, float]]: ...
def device_get(x: A) -> A: ...
def make_jaxpr(fn: Callable[[*As], Any]) -> Callable[[*As], ClosedJaxpr]: ...
def grad(f: Callable[[A, *Rest], float]) -> Callable[[A, *Rest], A]: ...

NumDevices = NewType("NumDevices", int)

def device_count() -> int: ...
def local_devices() -> list[Device]: ...
def devices(backend: Literal["cpu", "gpu", "tpu"] | None = None) -> list[Device]: ...
def device_put_replicated(x: A, devices: list[Device]) -> AuxDim[NumDevices, A]: ...
def device_put(x: A, device: Device | None = None) -> A: ...

class Device:
    platform: Literal["cpu", "gpu", "tpu"]

checkpoint_policies = SimpleNamespace(
    dots_with_no_batch_dims_saveable=...,
)

def checkpoint(
    fun: CallableT,
    prevent_cse: bool = True,
    static_argnums: tuple[int, ...] | None = None,
    *,
    policy: Callable[..., bool] | None = None,
) -> CallableT: ...
def block_until_ready(x: A) -> A: ...

class default_device:
    def __init__(self, device: Device): ...
    def __enter__(self): ...
    def __exit__(self, *args: Any): ...

class log_compiles:
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
