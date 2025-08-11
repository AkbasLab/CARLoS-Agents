from src.point import Point
from src.environment import Environment
import numpy as np
import src.sensor_detection as SD
from src.sensor import Sensor
from src.vehicle import Vehicle


class SensorArray:
    def __init__(
        self, num_sensors: int, sensor_length: float, sensor_angle_spread: float
    ):
        self.num_sensors = num_sensors
        self.sensor_angle_spread = sensor_angle_spread
        self.setup_sensors(sensor_length)

    def setup_sensors(self, sensor_length: float): 
        angle_max = self.sensor_angle_spread / 2
        angles = np.linspace(angle_max, -angle_max, self.num_sensors)
        self.sensors = [
            Sensor(sensor_length=sensor_length, angle_offset=angle) for angle in angles
        ]

    def update_sensors(
        self, origin_point: Point, direction_angle: float
    ):
        for sensor in self.sensors:
            sensor.update_sensor(origin_point, direction_angle)

    def sense(self, env: Environment, vehicle: Vehicle):
        self.update_sensors(
            origin_point=vehicle.center_point,
            direction_angle=vehicle.heading,
        )
        detection_points = []
        detection_distances = []
        for sensor in self.sensors:
            point, distance = SD.get_lane_detection(sensor=sensor, lane=env.lane)
            detection_points.append(point)
            detection_distances.append(distance)
        return detection_points, detection_distances
