# pylint: skip-file

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar, TypeVarTuple, overload, type_check_only

from jax.numpy import ndarray

A = TypeVar("A")
B = TypeVar("B")
Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)
Float2 = TypeVar("Float2", bound=float)

def tree_map_with_path(
    f: Callable[[KeyPath, ndarray[*Shape, Float]], Any],
    tree: A,
) -> Any: ...
@overload
def tree_map(
    f: Callable[[ndarray[*Shape, Float]], Any],
    tree: A,
    is_leaf: Callable[[Any], bool] | None = None,
) -> A: ...
@overload
def tree_map(
    f: Callable[[ndarray[*Shape, Any] | None, ndarray[*Shape, Any]], ndarray[*Shape, Any]],
    tree: A,
    rest: A,
    is_leaf: Callable[[Any], bool] = ...,
) -> A: ...
@overload
def tree_map(
    f: Callable[[ndarray[*Shape, Any], ndarray[*Shape, Any] | None], ndarray[*Shape, Float2]],
    tree: A,
    rest: A,
    is_leaf: Callable[[Any], bool] = ...,
) -> A: ...
def tree_all(tree: A) -> bool: ...
@dataclass
class GetAttrKey:
    name: str

@dataclass
class SequenceKey:
    idx: int

@dataclass
class DictKey:
    key: Hashable

KeyEntry: TypeAlias = GetAttrKey | SequenceKey | DictKey
KeyPath: TypeAlias = tuple[KeyEntry, ...]

class PyTreeDef(Generic[A]): ...

@type_check_only
class Leaves(Generic[A], list[ndarray[*tuple[Any, ...], float]]): ...

@type_check_only
class LeavesWithPath(Generic[A], list[tuple[KeyPath, ndarray[*tuple[Any, ...], float]]]): ...

@overload
def tree_leaves(tree: Any, is_leaf: Callable[[Any], bool]) -> Any: ...
@overload
def tree_leaves(tree: A, is_leaf: None = None) -> Leaves[A]: ...
def tree_leaves_with_path(tree: A) -> LeavesWithPath[A]: ...
def tree_structure(tree: A) -> PyTreeDef[A]: ...
def tree_unflatten(treedef: PyTreeDef[A], leaves: Leaves[A]) -> A: ...
def tree_flatten(tree: A) -> tuple[Leaves[A], PyTreeDef[A]]: ...
