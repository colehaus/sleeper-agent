# pylint: skip-file

from typing import Literal, TypeAlias, TypeVar, TypeVarTuple, overload

from numpy import Fin, ndarray

Shape = TypeVarTuple("Shape")
DType = TypeVar("DType")
Dim1 = TypeVar("Dim1", bound=int)
Int = TypeVar("Int", bound=int)

Two: TypeAlias = Literal[2]
Num = TypeVar("Num", bound=int)

KeyArray: TypeAlias = ndarray[Two, int]

def PRNGKey(seed: int | ndarray[int]) -> ndarray[Two, int]: ...
@overload
def split(key: KeyArray) -> ndarray[Two, Two, int]: ...
@overload
def split(key: KeyArray, num: Num) -> ndarray[Num, Two, int]: ...
@overload
def normal(key: KeyArray, shape: tuple[*Shape], dtype: type[DType] = float) -> ndarray[*Shape, DType]: ...
@overload
def normal(key: KeyArray) -> float: ...
@overload
def uniform(
    key: KeyArray, shape: tuple[*Shape], minval: float = 0.0, maxval: float = 1.0, dtype: type[DType] = float
) -> ndarray[*Shape, DType]: ...
@overload
def uniform(key: KeyArray) -> float: ...
def randint(
    key: KeyArray, shape: tuple[*Shape], minval: int, maxval: int, dtype: type[DType] = int
) -> ndarray[*Shape, DType]: ...
def bernoulli(key: KeyArray, shape: tuple[*Shape]) -> ndarray[*Shape, bool]: ...
def multivariate_normal(
    key: KeyArray, mean: ndarray[Dim1, DType], cov: ndarray[Dim1, Dim1, DType]
) -> ndarray[Dim1, DType]: ...
@overload
def choice(
    key: KeyArray, a: ndarray[Dim1, DType], shape: tuple[*Shape], replace: bool = True
) -> ndarray[*Shape, DType]: ...
@overload
def choice(key: KeyArray, a: Int, p: ndarray[Int, float]) -> ndarray[Fin[Int]]: ...
def permutation(key: KeyArray, x: ndarray[*Shape, A], independent: bool = False) -> ndarray[*Shape, A]: ...

A = TypeVar("A")
