import math
from typing import List, Tuple


class Value:
    def __init__(
        self,
        data: float | int = 0.0,
        _children: tuple = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev: set = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Value":  # self * other
        return self * other

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __neg__(self) -> "Value":
        return self * -1

    def __pow__(self, other) -> "Value":
        assert isinstance(other, (int, float)), "only support int or float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other) -> "Value":
        return self * (other**-1)

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
