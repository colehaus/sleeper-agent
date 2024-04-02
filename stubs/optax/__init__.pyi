# pylint: skip-file

from typing import Any, Callable, Generic, NamedTuple, Protocol, TypeVar, TypeVarTuple, type_check_only

import numpy as np
from equinox import Grads
from jax.numpy import ndarray

NumClasses = TypeVar("NumClasses", bound=int)
Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)

@type_check_only
class ArraysOf(Generic[A]): ...

A = TypeVar("A")

class OptState(Generic[A, Float]): ...

class FactoredState(NamedTuple):
    count: ndarray[int]  # pyright: ignore[reportIncompatibleMethodOverride]
    v_row: Any
    v_col: Any
    v: Any

class GradientTransformation(NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn

class TransformInitFn(Protocol):
    def __call__(self, params: ArraysOf[A]) -> OptState[A, np.float32]: ...

class TransformUpdateFn(Protocol):
    def __call__(
        self, updates: Grads[A], state: OptState[A, Float], params: A | None = None
    ) -> tuple[Grads[A], OptState[A, Float]]: ...

def adafactor(learning_rate: float) -> GradientTransformation: ...
def adam(learning_rate: float) -> GradientTransformation: ...
def sgd(learning_rate: float) -> GradientTransformation: ...
def rmsprop(learning_rate: float) -> GradientTransformation: ...
def adamw(
    learning_rate: float | Callable[[int], float], weight_decay: float = 0.0001
) -> GradientTransformation: ...
def clip_by_global_norm(max_norm: float) -> GradientTransformation: ...
def apply_every(k: int) -> GradientTransformation: ...
def scale(step_size: float) -> GradientTransformation: ...
def chain(
    *transforms: GradientTransformation,
) -> GradientTransformation: ...
def softmax_cross_entropy_with_integer_labels(
    logits: ndarray[*Shape, NumClasses, Float], labels: ndarray[*Shape, np.Fin[NumClasses]]
) -> ndarray[*Shape, Float]: ...
def softmax_cross_entropy(
    logits: ndarray[*Shape, NumClasses, Float], labels: ndarray[*Shape, NumClasses, Float]
) -> ndarray[*Shape, Float]: ...
def sigmoid_binary_cross_entropy(
    logits: ndarray[*Shape, Float], labels: ndarray[*Shape, bool]
) -> ndarray[*Shape, Float]: ...
def warmup_cosine_decay_schedule(
    init_value: float, peak_value: float, warmup_steps: int, decay_steps: int
) -> Callable[[int], float]: ...
def apply_if_finite(inner: GradientTransformation, max_consecutive_errors: int) -> GradientTransformation: ...
