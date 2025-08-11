from src.point import Point
import math

class Sensor:
    def __init__(
        self, sensor_length: float, angle_offset: float
    ):  
        self.sensor_length = sensor_length
        self.angle_offset = angle_offset

    def update_sensor(
        self, origin_point: Point, direction_angle: float
    ):  
        self.origin_point = origin_point
        self.end_point = self.calculate_end_point(direction_angle)

    def calculate_end_point(
        self, direction_angle: float
    ) -> Point: 
        new_angle = direction_angle + self.angle_offset
        new_x = self.origin_point.x + self.sensor_length * math.cos(new_angle)
        new_y = self.origin_point.y + self.sensor_length * math.sin(new_angle)

        return Point(new_x, new_y) 
    
