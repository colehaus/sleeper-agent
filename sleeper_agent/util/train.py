from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Literal, NamedTuple, Protocol, TypeVar, TypeVarTuple

import equinox as eqx
import jax
import optax
from numpy import float32, ndarray
from tqdm.auto import tqdm

Batch = TypeVar("Batch")
Input = TypeVar("Input")
Extra = TypeVar("Extra")
Float = TypeVar("Float", bound=float)
Model = TypeVar("Model", bound=eqx.Module)
Args = TypeVarTuple("Args")
Shape = TypeVarTuple("Shape")
A = TypeVar("A")


class StepReturn(NamedTuple, Generic[Model, Float, Extra]):
    model: Model
    target_loss: ndarray[Float]
    other_return: Extra
    opt_state: optax.OptState[Model, float32]
    new_key: jax.Array


class StepFn(Protocol[Model, Input, *Args, Float, Extra]):
    def __call__(
        self,
        model: Model,
        opt_state: optax.OptState[Model, float32],
        input_: Input,
        *args: *Args,
        key: jax.Array,
    ) -> StepReturn[Model, Float, Extra]:
        ...


def mk_step(
    update_fn: optax.TransformUpdateFn,
    loss_fn: Callable[[Model, Input, *Args, jax.Array], tuple[tuple[ndarray[Float], Extra], eqx.Grads[Model]]],
) -> StepFn[Model, Input, *Args, Float, Extra]:
    @eqx.filter_jit
    def step(
        model: Model,
        opt_state: optax.OptState[Model, float32],
        input_: Input,
        *args: *Args,
        key: jax.Array,
    ):
        key, new_key = jax.random.split(key, num=2)
        (loss, extra), grads = loss_fn(model, input_, *args, key)
        updates, opt_state = update_fn(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return StepReturn(model, loss, extra, opt_state, new_key)

    return step


def mk_stop_fn(
    fn: Callable[[Sequence[A]], Literal["continue", "stop"]],
    extractor: Callable[[Float, Extra], A],
    *,
    lookback_len: int,
) -> Callable[[Float, Extra], Literal["continue", "stop"]]:
    losses = deque[A](maxlen=lookback_len)

    def inner(loss: Float, extra: Extra):
        loss_ = extractor(loss, extra)
        losses.append(loss_)
        return "stop" if len(losses) == lookback_len and fn(losses) == "stop" else "continue"

    return inner


def train(  # noqa: PLR0913
    model: Model,
    opt_state: optax.OptState[Model, float32],
    step: StepFn[Model, Batch, Float, Extra],
    batch_fn: Callable[[], Batch],
    postfix_fn: Callable[[Float, Extra], Mapping[str, Any]],
    stop_fn: Callable[[Float, Extra], Literal["continue", "stop"]],
    callbacks: Sequence[Callable[[Float], None]] = [],
    *,
    key: jax.Array,
):
    tqdm_bar = tqdm(unit="batch")
    continue_ = "continue"
    while continue_ == "continue":
        batch = batch_fn()
        model, loss, extra, opt_state, key = step(
            model,
            opt_state,
            batch,
            key=key,
        )
        tqdm_bar.set_postfix(**postfix_fn(loss.item(), extra))
        tqdm_bar.update()
        for callback in callbacks:
            callback(loss.item())
        continue_ = stop_fn(loss.item(), extra)
    return model, opt_state
