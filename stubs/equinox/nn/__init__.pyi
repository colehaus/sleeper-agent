# pylint: skip-file

from typing import Any, Callable, Generic, Literal, TypeVar, TypeVarTuple, overload

from jax.numpy import ndarray
from jax.random import KeyArray
from numpy import Fin

from .._module import Module
from . import _shared as _shared

InDim = TypeVar("InDim", bound=int)
HiddenDim = TypeVar("HiddenDim", bound=int)
OutDim = TypeVar("OutDim", bound=int)

# Note that that `Float` parameter used in most of these declarations is not quite right.
# Most of them are actually `float32` after initialization
# (e.g. a `Linear` layer initializes the weights and biases as `float32` arrays).
# But we can change this after the fact with by `tree_map`ing `.astype`.
# Including `Float` as a parameter is a small lie then, but it allows us to ensure
# that all our layers fit together correctly with the same floating point type which
# is a more valuable property for the types to guarantee.
# e.g. It ensures that any `ndarray`s we declare inline (`np.ones`, etc.) use the right type.

class Linear(Module, Generic[InDim, OutDim, Float]):
    bias: ndarray[OutDim, Float]
    weight: ndarray[OutDim, InDim, Float]
    def __init__(
        self,
        in_features: InDim | Literal["scalar"],
        out_features: OutDim | Literal["scalar"],
        use_bias: bool = True,
        *,
        key: KeyArray,
    ) -> None: ...
    def __call__(self, x: ndarray[InDim, Float]) -> ndarray[OutDim, Float]: ...

class GRUCell(Module, Generic[InDim, HiddenDim]):
    def __init__(self, input_size: InDim, hidden_size: HiddenDim, *, key: KeyArray) -> None: ...
    def __call__(
        self, input: ndarray[InDim, Float], hidden: ndarray[HiddenDim, Float]
    ) -> ndarray[HiddenDim, Float]: ...

SpatialDims = TypeVar("SpatialDims", bound=int)
Shape = TypeVarTuple("Shape")

InChannels = TypeVar("InChannels", bound=int)
OutChannels = TypeVar("OutChannels", bound=int)
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)

class Conv(Module, Generic[SpatialDims, InChannels, OutChannels]):
    def __init__(
        self,
        num_spatial_dims: SpatialDims,
        in_channels: InChannels,
        out_channels: OutChannels,
        kernel_size: int,
        *,
        key: KeyArray,
    ) -> None: ...
    def __call__(self, x: ndarray[InChannels, *Shape, Float]) -> ndarray[OutChannels, *Shape, Float]: ...

class Conv2d(Conv[Literal[2], InChannels, OutChannels]):
    def __init__(
        self,
        in_channels: InChannels,
        out_channels: OutChannels,
        kernel_size: int,
        *,
        key: KeyArray,
    ) -> None: ...
    def __call__(self, x: ndarray[InChannels, Dim1, Dim2, Float]) -> ndarray[OutChannels, Dim1, Dim2, Float]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class Pool(Module): ...

class MaxPool2d(Pool):
    def __init__(self, kernel_size: int, stride: int = 1) -> None: ...
    def __call__(self, x: ndarray[InChannels, Any, Any, Float]) -> ndarray[InChannels, Any, Any, Float]: ...

EmbedDim = TypeVar("EmbedDim", bound=int)
VocabSize = TypeVar("VocabSize", bound=int)

class Embedding(Module, Generic[VocabSize, EmbedDim, Float]):
    num_embeddings: VocabSize
    embedding_size: EmbedDim
    weight: ndarray[VocabSize, EmbedDim, Float]
    def __init__(self, num_embeddings: VocabSize, embedding_size: EmbedDim, *, key: KeyArray) -> None: ...
    def __call__(self, x: ndarray[Fin[VocabSize]]) -> ndarray[EmbedDim, Float]: ...

class Dropout(Module, Generic[Float]):
    def __init__(self, p: Float = 0.5, inference: bool = False) -> None: ...
    def __call__(
        self, x: ndarray[*Shape, Float], key: KeyArray | None = None, inference: bool | None = None
    ) -> ndarray[*Shape, Float]: ...

class LayerNorm(Module, Generic[EmbedDim, Float]):
    def __init__(
        self, shape: EmbedDim, eps: Float = ..., use_weight: bool = True, use_bias: bool = True
    ) -> None: ...
    def __call__(self, x: ndarray[EmbedDim, Float]) -> ndarray[EmbedDim, Float]: ...

QSeqLength = TypeVar("QSeqLength", bound=int)
KVSeqLength = TypeVar("KVSeqLength", bound=int)
QuerySize = TypeVar("QuerySize", bound=int)
KeySize = TypeVar("KeySize", bound=int)
ValueSize = TypeVar("ValueSize", bound=int)
OutputSize = TypeVar("OutputSize", bound=int)
NumHeads = TypeVar("NumHeads", bound=int)
Float = TypeVar("Float", bound=float)

class MultiheadAttention(Module, Generic[NumHeads, QuerySize, KeySize, ValueSize, OutputSize, Float]):
    @overload
    def __init__(
        self: MultiheadAttention[NumHeads, QuerySize, QuerySize, QuerySize, QuerySize, Float],
        num_heads: NumHeads,
        query_size: QuerySize,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: Float = 0.0,
        *,
        key: KeyArray,
    ) -> None: ...
    @overload
    def __init__(
        self,
        num_heads: NumHeads,
        query_size: QuerySize,
        key_size: KeySize,
        value_size: ValueSize,
        output_size: OutputSize,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: Float = 0.0,
        *,
        key: KeyArray,
    ) -> None: ...
    def __call__(
        self,
        query: ndarray[QSeqLength, QuerySize, Float],
        key_: ndarray[KVSeqLength, KeySize, Float],
        value: ndarray[KVSeqLength, ValueSize, Float],
        mask: ndarray[NumHeads, QSeqLength, KVSeqLength, bool]
        | ndarray[QSeqLength, KVSeqLength, bool]
        | None = None,
        *,
        key: KeyArray | None = None,
        inference: bool | None = None,
    ) -> ndarray[QSeqLength, OutputSize, Float]: ...

A = TypeVar("A")

class Shared(Module, Generic[A]):
    pytree: A
    def __init__(self, pytree: A, where: Callable[[A], Any], get: Callable[[A], Any]) -> None: ...
    def __call__(self) -> A: ...
