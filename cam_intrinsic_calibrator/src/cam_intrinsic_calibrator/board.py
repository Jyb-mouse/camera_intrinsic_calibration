class ChessBoard:
    def __init__(self, cb_shape, square_size):
        self._cb_shape = cb_shape
        self._square_size = square_size

    @property
    def cb_shape(self):
        return self._cb_shape

    @property
    def square_size(self):
        return self._square_size


class RingBoard:
    def __init__(self, ring_shape, ring_distance):
        self._ring_shape = ring_shape
        self._ring_distance = ring_distance

    @property
    def ring_shape(self):
        return self._ring_shape

    @property
    def ring_distance(self):
        return self._ring_distance
