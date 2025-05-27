import math
from typing import List, TypeVar, Generic
from qnnum import Qnnum

T = TypeVar('T',float, Qnnum)

class Point(Generic[T]):
    def __init__(self, x: T = None, y: T = None):
        self.x = x
        self.y = y

    def __eq__(self, other: 'Point[T]') -> bool:
        return self.x == other.x and self.y == other.y

    def distance_squared(self, a: 'Point[T]') -> T:
        return (self.x - a.x) ** 2 + (self.y - a.y) ** 2

    def finite(self) -> bool:
        return self.x is not None and self.y is not None

    @staticmethod
    def slope(a: 'Point[T]', b: 'Point[T]') -> T:
        if b.x - a.x == 0:
            raise ValueError("Slope is undefined for vertical line.")
        return (b.y - a.y) / (b.x - a.x)

    @staticmethod
    def midpoint(a: 'Point[T]', b: 'Point[T]') -> 'Point[T]':
        return Point((a.x + b.x) / 2, (a.y + b.y) / 2)

class Edge(Generic[T]):
    def __init__(self, a: Point[T], b: Point[T]):
        self.a = a
        self.b = b

class Circle(Generic[T]):
    def __init__(self, center: Point[T], radius: T):
        self.center = center
        self.radius = radius

    def contains(self, p: Point[T]) -> bool:
        return self.center.distance_squared(p) <= self.radius ** 2

    def infinite(self) -> bool:
        return self.radius is None

class Triangle(Generic[T]):
    def __init__(self, a: Point[T] = None, b: Point[T] = None, c: Point[T] = None):
        self.a = a
        self.b = b
        self.c = c

    def valid(self) -> bool:
        return self.a is not None and self.b is not None and self.c is not None

    def has_vertex(self, p: Point[T]) -> bool:
        return self.a == p or self.b == p or self.c == p

    def circumcircle(self) -> Circle[T]:
        # Implementation of circumcircle calculation would go here
        pass

    def has_edge(self, e: Edge[T]) -> bool:
        return (self.a == e.a and self.b == e.b) or (self.a == e.b and self.b == e.a) or \
               (self.a == e.a and self.c == e.b) or (self.a == e.b and self.c == e.a) or \
               (self.b == e.a and self.c == e.b) or (self.b == e.b and self.c == e.a)

    def edges(self) -> List[Edge[T]]:
        return [Edge(self.a, self.b), Edge(self.b, self.c), Edge(self.c, self.a)]

    def __eq__(self, other: 'Triangle[T]') -> bool:
        return (self.a == other.a and self.b == other.b and self.c == other.c) or \
               (self.a == other.b and self.b == other.c and self.c == other.a) or \
               (self.a == other.c and self.b == other.a and self.c == other.b)

def abs_value(a: T) -> T:
    return abs(a)

# Specialization for qnnum.Qnnum would be handled separately

