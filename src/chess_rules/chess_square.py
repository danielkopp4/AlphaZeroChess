from chess import Square as CSquare
from chess import square as csquare

eps = 1E-8

class ChessSquare:
    def __init__(self, x: int, y: int):
        self._invalid = x < 0 or y < 0 or x > 7 or y > 7
        self._x = x
        self._y = y
        self._csquare = csquare(self._x, self._y)

    @property
    def chess_square(self) -> CSquare:
        return self._csquare

    @property
    def invalid(self) -> bool:
        return self._invalid

    def __add__(self, other) -> 'ChessSquare':
        if type(other) == int:
            if other == 0:
                return self

            return ChessSquare(self._x + other, self._y + other)

        
        if type(other) == tuple or type(other) == list:
            if len(other) != 2:
                raise Exception("must be length 2 to add to square")
            
            if type(other[0]) != int or type(other[1]) != int:
                raise Exception("must be an iterable of ints")

            return ChessSquare(self._x + other[0], self._y + other[1])

        if type(other) == ChessSquare:
            return ChessSquare(self._x + other._x, self._y + other._y)

        raise Exception("unsupported addition between '{}' and '{}'".format(type(self), type(other)))

    def __radd__(self, other) -> 'ChessSquare':
        return self + other

    def __sub__(self, other) -> 'ChessSquare':
        return self + (-other)

    def __rsub__(self, other) -> 'ChessSquare':
        return other + (-self)

    def __neg__(self) -> 'ChessSquare':
        return ChessSquare(-self._x, -self._y)

    def __mul__(self, scalar) -> 'ChessSquare':
        if type(scalar) != int:
            raise Exception("scalar type '{}' not supported".format(type(scalar)))

        return ChessSquare(scalar * self._x, scalar * self._y)

    def __rmul__(self, scalar) -> 'ChessSquare':
        return self * scalar

    def __str__(self) -> str:
        return "({},{})".format(self._x, self._y)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: 'ChessSquare') -> bool:
        if self is other:
            return True

        if type(other) != ChessSquare:
            return False

        return abs(self._x - other._x) < eps and abs(self._y - other._y) < eps

    def __abs__(self) -> 'ChessSquare':
        return ChessSquare(abs(self._x), abs(self._y))

    def __floordiv__(self, scalar: int) -> 'ChessSquare':
        return ChessSquare(self.x // scalar, self.y // scalar)


    def __truediv__(self, scalar: float) -> 'ChessSquare':
        return ChessSquare(self.x / scalar, self.y / scalar)

    def slope(self) -> float:
        if self._x == 0:
            if self._y > 0:
                return float('inf')

            return -float('inf')

        return self._y / self._x

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @staticmethod
    def from_CSquare(csquare: CSquare) -> 'ChessSquare':
        x = csquare % 8
        y = csquare // 8
        return ChessSquare(x, y)

    def color_flip(self) -> 'ChessSquare':
        return ChessSquare(self.x, 7 - self.y)
