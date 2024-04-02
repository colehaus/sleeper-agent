from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, TypeVarTuple, cast

import equinox as eqx
import jax
from jax.numpy import ndarray

if TYPE_CHECKING:
    from optax import ArraysOf

A = TypeVar("A")
B = TypeVar("B")
Dim1 = TypeVar("Dim1", bound=int)
Num = TypeVar("Num", bound=int)
NumLayers = TypeVar("NumLayers", bound=int)
Rest = TypeVarTuple("Rest")

KeyArray: TypeAlias = "ndarray[Literal[2], int]"


def strip_part(x: eqx.PartOf[A]) -> A:
    return cast(A, x)


def split_optional(key: KeyArray | None, num: Num) -> ndarray[Num, Literal[2], int] | Sequence[None]:
    return [None] * num if key is None else jax.random.split(key, num)


def scan_layers(
    initial: A,
    initial_key: KeyArray | None,
    layers: jax.AuxDim[NumLayers, Callable[[A, *Rest, KeyArray | None], A]],
    # Constant across layers
    *args: *Rest,
) -> A:
    return scan_layers_with_intermediates(initial, initial_key, layers, *args)[0]


def _aux_part_of(x: eqx.PartOf[jax.AuxDim[NumLayers, B]]) -> jax.AuxDim[NumLayers, eqx.PartOf[B]]:
    return cast(Any, x)


def scan_layers_with_intermediates(
    initial: A,
    initial_key: KeyArray | None,
    layers: jax.AuxDim[NumLayers, Callable[[A, *Rest, KeyArray | None], A]],
    # Constant across layers
    *args: *Rest,
) -> tuple[A, jax.AuxDim[NumLayers, A]]:
    dynamic_layers_, static_layers_ = eqx.partition(layers, lambda x: eqx.is_array(x) and x.ndim > 0)
    dynamic_layers, static_layers = _aux_part_of(dynamic_layers_), _aux_part_of(static_layers_)

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#practical-notes
    @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def f(
        acc: tuple[A, KeyArray | None],
        dynamic_layer: eqx.PartOf[Callable[[A, *Rest, KeyArray | None], A]],
    ):
        old_x, acc_key = acc
        layer_key, acc_key = split_optional(acc_key, num=2)
        layer = eqx.combine(dynamic_layer, drop_aux_dim(static_layers))
        new_x = layer(old_x, *args, layer_key)
        return (new_x, layer_key), new_x

    (x, _), layer_outs = jax.lax.scan(f, (initial, initial_key), dynamic_layers)
    return x, layer_outs


def drop_aux_dim(x: jax.AuxDim[Dim1, A]) -> A:
    return x  # pyright: ignore


def arrays_of(x: A) -> ArraysOf[A]:
    return cast(Any, eqx.filter(x, eqx.is_array))
