import numpy as np
import torch


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def norm(self, p: int = 2) -> float:
        return (self.x**p + self.y**p) ** (1.0 / p)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def __mul__(self, other: float) -> "Point":
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: float) -> "Point":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Point":
        return self.__mul__(1.0 / other)

    def values(self):
        return [self.x, self.y]

    def distanceTo(self, other: "Point") -> float:
        return (self - other).norm(p=2)

    def rotate_point_by_radians(self, point: "Point", rotation_angle: float) -> "Point":
        rotation_angle = np.clip(float(rotation_angle), -np.pi, np.pi)
        
        x1, y1 = point.x, point.y
        x2, y2 = self.x, self.y
        dx = x2 - x1
        dy = y2 - y1

        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)

        new_dx = cos_angle * dx - sin_angle * dy
        new_dy = sin_angle * dx + cos_angle * dy

        offset_x = x1 + new_dx
        offset_y = y1 + new_dy

        return Point(offset_x, offset_y)
    
