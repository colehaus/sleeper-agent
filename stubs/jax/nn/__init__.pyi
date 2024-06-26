# pylint: skip-file

from typing import TypeVar, TypeVarTuple

from numpy import ndarray

from . import initializers as initializers

Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)

def relu(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def sigmoid(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def log_softmax(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def softmax(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def gelu(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def log_sigmoid(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
