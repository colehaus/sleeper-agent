from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, TypeVarTuple, cast, overload

from jax.numpy import ndarray

if TYPE_CHECKING:
    from numpy import Fin, Sum

A = TypeVar("A")
Int = TypeVar("Int", bound=int)
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)
DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")
Label = TypeVar("Label", bound=str)


def sum_(operands: tuple[*Shape]) -> Sum[*Shape]:
    return cast(Any, sum(cast(tuple[int, ...], operands)))


class fin(int, Generic[Int]):  # noqa: N801
    def __new__(cls, x: int | ndarray[int], max_: Int) -> Fin[Int]:
        assert 0 <= x < max_
        return cast(Any, x)


class InstanceSingleton(int, Generic[Label]):
    """As with `Singleton`, but we ensure that there's only one value with the label for a given instance.
    This is primarily used for defining "internal type variables".
    i.e. It's sometimes the case that the implementation of a class requires an annotating type
    (for e.g. an array dimension).
    But the class's user is uninterested in this type so we don't want to "pollute" the class with a type variable.
    (If we simply declare the type variable inside the class, pyright basically treats it as `Any`.)
    So instead we create an `InstanceSingleton` as the annotating type—it still gives us the guarantee
    a type variable would that any two arrays with this type for a dimension have the same size at runtime.
    """

    history: dict[tuple[int, str], int] = {}  # noqa: RUF012

    def __new__(cls, instance: Any, label: Label, value: int) -> InstanceSingleton[Label]:
        match cls.history.get((id(instance), label)):
            case None:
                cls.history[(id(instance), label)] = value
                return cast(Any, value)
            case v:
                assert v == value, (instance, label, v, value)
                return cast(Any, v)


class declare_axis(Generic[A]):  # noqa: N801
    @overload
    def __new__(cls, axis: Literal[0], array: ndarray[Any, *Shape, DType]) -> ndarray[A, *Shape, DType]:
        ...

    @overload
    def __new__(
        cls, axis: Literal[1], array: ndarray[Dim1, Any, *Shape, DType]
    ) -> ndarray[Dim1, A, *Shape, DType]:
        ...

    @overload
    def __new__(
        cls, axis: Literal[2], array: ndarray[Dim1, Dim2, Any, *Shape, DType]
    ) -> ndarray[Dim1, Dim2, A, *Shape, DType]:
        ...

    def __new__(cls, axis: int, array: ndarray[*Shape, DType]) -> ndarray[*Shape, DType]:
        return array


def list_dicts_to_dict_lists(
    dicts: Sequence[Mapping[A, Any]],
) -> dict[A, list[Any]]:
    return {k: [d[k] for d in dicts] for k in dicts[0]}
