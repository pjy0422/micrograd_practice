import math
from typing import List, Tuple


class Value:
    def __init__(
        self, data: float = 0.0, _children: tuple = (), _op: str = "", label: str = ""
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
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo : List[Value] = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
        
        build_topo(self)
        
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()
        