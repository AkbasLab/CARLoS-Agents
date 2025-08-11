import torch
import math
import numpy as np 
from src.point import Point

class Vehicle:

    mph_to_fps_conversion = 5280 / 3600
    fps_to_mph_conversion = 3600 / 5280
    mph2_to_fps2_conversion = 5280 / (3600**2)
    fps2_to_mph2_conversion = (3600**2) / 5280

    def __init__(
        self,
        vehicle_length_ft: float = 12,
        vehicle_width_ft: float = 6,
        min_speed_mph: float = 0.0,
        max_speed_mph: float = 150.0,
        sec_0_to_60: float = 8.0,
        ):  
        self.body = VehicleBody(
            vehicle_length=vehicle_length_ft, vehicle_width=vehicle_width_ft
        )
        self.min_speed_fps = self.mph_to_fps(min_speed_mph)
        self.max_speed_fps = self.mph_to_fps(max_speed_mph)
        self.max_acceleration_fps2 = self.mph_to_fps(60.0) / sec_0_to_60
        self.max_breaking_fps2 = 15.0

    def vehicle_setup(
        self, center_point: Point, heading: float, speed_mph: float
        ):  
        self.center_point = center_point
        speed_fps = self.mph_to_fps(speed_mph)
        self.speed_fps = np.clip(speed_fps, self.min_speed_fps, self.max_speed_fps)
        self.heading = heading
        self.distance_travelled_ft = 0
        self.acceleration_fps2 = 0
        self.body.build_body(center_point=center_point, turn_angle=heading)

    def vehicle_capabilities_str(self): 
        return (
            f"Vehicle Capabilities: "
            f"Speed: {self.fps_to_mph(self.min_speed_fps)} to {self.fps_to_mph(self.max_speed_fps)} mph -- "
            f"Max Acceleration: {self.max_acceleration_fps2} fps^2 = {self.fps2_to_mph2(self.max_acceleration_fps2)} mph^2 -- "
            f"Max Breaking: {self.max_breaking_fps2} fps^2 = {self.fps2_to_mph2(self.max_breaking_fps2)} mph^2"
        )

    def vehicle_state_str(self):  
        return (
            f"Center: ({self.center_point.x}, {self.center_point.y}) -- "
            f"Heading: ({self.heading * 180 / np.pi}) deg -- "
            f"Speed: {self.speed_mph} mph -- "
            f"Acceleration: {self.acceleration_mph2} mph^2 -- "
            f"Distance travelled: {self.distance_travelled_ft} feet"
        )

    @property
    def speed_mph(self):  
        return self.fps_to_mph(self.speed_fps)

    @property
    def acceleration_mph2(self):  
        return self.fps2_to_mph2(self.acceleration_fps2)

    @property
    def distance_travelled_miles(self):  
        return self.distance_travelled_ft / 5280

    def mph_to_fps(self, value_mph: float):  
        return value_mph * self.mph_to_fps_conversion

    def fps_to_mph(self, value_fps: float): 
        return value_fps * self.fps_to_mph_conversion

    def mph2_to_fps2(self, value_mph2: float):  
        return value_mph2 * self.mph2_to_fps2_conversion

    def fps2_to_mph2(self, value_fps2: float):  
        return value_fps2 * self.fps2_to_mph2_conversion

    def get_direction(self, angle: np.float32 = None) -> torch.Tensor:
        angle = angle or self.heading
        return torch.tensor([np.cos(float(angle)), np.sin(float(angle))])

    def get_heading_point(self, angle: float = None) -> Point:
        angle = angle or self.heading
        heading_direction = np.array(self.get_direction(angle))
        hx = self.center_point.x + heading_direction[0]
        hy = self.center_point.y + heading_direction[1]
        return Point(hx, hy)

    def update_position(
        self, steering_rad: float, acceleration_mph2: float, dt_sec: float
        ):  
        self.acceleration_fps2 = np.clip(
            self.mph2_to_fps2(float(acceleration_mph2)),
            -float(self.max_breaking_fps2),
            float(self.max_acceleration_fps2),
        )

        new_speed = self.speed_fps + (self.acceleration_fps2 * dt_sec)
        new_speed = np.clip(new_speed, self.min_speed_fps, self.max_speed_fps)

        turn_angle = steering_rad * dt_sec
        new_heading = self.heading + turn_angle
        new_direction = self.get_direction(new_heading)

        distance = (self.speed_fps * dt_sec) + (
            0.5 * self.acceleration_fps2 * (dt_sec**2)
        )

        cx = self.center_point.x + new_direction[0] * distance
        cy = self.center_point.y + new_direction[1] * distance
        self.center_point = Point(cx, cy)

        self.heading = new_heading
        self.speed_fps = new_speed
        self.distance_travelled_ft += distance
        self.body.build_body(self.center_point, self.heading)

class VehicleBody:
    def __init__(self, vehicle_length: float, vehicle_width: float):
        self.length = vehicle_length
        self.width = vehicle_width
        self.set_base_corners()

    def set_base_corners(self):
        half_length = self.length / 2
        half_width = self.width / 2
        self.base_corners = [
            Point(half_length, -half_width),  
            Point(half_length, half_width),  
            Point(-half_length, half_width),  
            Point(-half_length, -half_width),  
        ]

    def build_body(self, center_point: Point, turn_angle: float):
        self.corners = []
        for c in self.base_corners:
            c = c + center_point
            c = c.rotate_point_by_radians(center_point, turn_angle)
            self.corners.append(c)
