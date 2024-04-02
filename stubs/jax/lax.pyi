# pylint: skip-file

from collections.abc import Sequence
from typing import Callable, Hashable, TypeVar, TypeVarTuple, overload

from jax import AuxDim
from jax.numpy import ndarray

A = TypeVar("A")
B = TypeVar("B")
Out = TypeVar("Out")
Carry = TypeVar("Carry")
Xs = TypeVarTuple("Xs")
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
Dim1 = TypeVar("Dim1")
DType = TypeVar("DType")
Float = TypeVar("Float", bound=float)
Args = TypeVarTuple("Args")
Int = TypeVar("Int", bound=int)

def cond(
    pred: bool | ndarray[bool],
    true: Callable[[*Xs], Out],
    false: Callable[[*Xs], Out],
    *args: *Xs,
) -> Out: ...
def while_loop(
    cond_fun: Callable[[Carry], bool],
    body_fun: Callable[[Carry], Carry],
    init_val: Carry,
) -> Carry: ...
@overload
def scan(
    fun: Callable[[Carry, ndarray[*Shape, float]], tuple[Carry, None]],
    init: Carry,
    x: ndarray[Dim1, *Shape, float],
) -> tuple[Carry, None]: ...
@overload
def scan(
    fun: Callable[[Carry, A], tuple[Carry, B]],
    init: Carry,
    x: AuxDim[Dim1, A],
) -> tuple[Carry, AuxDim[Dim1, B]]: ...
def pmean(x: A, axis_name: Hashable) -> A: ...
def rsqrt(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def stop_gradient(x: A) -> A: ...
def switch(index: ndarray[Int], branches: Sequence[Callable[[*Args], B]], *operands: *Args) -> B: ...
