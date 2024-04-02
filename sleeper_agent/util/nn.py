from __future__ import annotations

from typing import Generic, Literal, TypeAlias, TypeVar, TypeVarTuple

import equinox as eqx
import jax
from numpy import ndarray

from sleeper_agent.util.jax import scan_layers
from sleeper_agent.util.misc import InstanceSingleton

Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)
InDim = TypeVar("InDim", bound=int)
OutDim = TypeVar("OutDim", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
Dim1 = TypeVar("Dim1", bound=int)


class FeedForward(eqx.Module, Generic[InDim, OutDim, Float]):
    linear: eqx.nn.Linear[InDim, OutDim, Float]
    norm: eqx.nn.LayerNorm[OutDim, Float]

    def __init__(self, *, in_features: InDim, out_features: OutDim, key: jax.Array):
        self.linear = eqx.nn.Linear(in_features=in_features, out_features=out_features, key=key)
        self.norm = eqx.nn.LayerNorm(out_features)

    def __call__(self, x: ndarray[InDim, Float], key: jax.Array | None) -> ndarray[OutDim, Float]:
        return self.norm(jax.nn.relu(self.linear(x)))


class FFStack(eqx.Module, Generic[Dim1, Float]):
    stack: jax.AuxDim[int, FeedForward[Dim1, Dim1, Float]]

    def __init__(self, *, dim: Dim1, num_layers: int, key: jax.Array):
        def mk_layer(
            layer_key: jax.Array,
        ) -> FeedForward[Dim1, Dim1, Float]:
            return FeedForward(in_features=dim, out_features=dim, key=layer_key)

        self.stack = eqx.filter_vmap(mk_layer)(jax.random.split(key, num=num_layers))

    def __call__(self, x: ndarray[Dim1, Float]) -> ndarray[Dim1, Float]:
        return scan_layers(x, None, self.stack)


class MLP(eqx.Module, Generic[InDim, OutDim, Float]):
    """Differs from `eqx.nn.MLP` in that it our `FeedForward` has layer norm and we use `scan`"""

    _EmbedDim: TypeAlias = InstanceSingleton[Literal["EmbedDim"]]

    initial: FeedForward[InDim, _EmbedDim, Float]
    stack: FFStack[_EmbedDim, Float]
    out: eqx.nn.Linear[_EmbedDim, OutDim, Float]

    def __init__(  # noqa: PLR0913
        self, *, in_dim: InDim, embed_dim: EmbedDim, out_dim: OutDim, num_hidden_layers: int, key: jax.Array
    ):
        init_key, stack_key, out_key = jax.random.split(key, num=3)
        embed_size_ = InstanceSingleton[Literal["EmbedDim"]](self, "EmbedDim", embed_dim)
        self.initial = FeedForward(in_features=in_dim, out_features=embed_size_, key=init_key)
        self.stack = FFStack(dim=embed_size_, num_layers=num_hidden_layers, key=stack_key)
        self.out = eqx.nn.Linear(in_features=embed_size_, out_features=out_dim, key=out_key)

    def __call__(self, x: ndarray[InDim, Float]) -> ndarray[OutDim, Float]:
        init_out = self.initial(x, key=None)
        stack_out = self.stack(init_out)
        return self.out(stack_out)
