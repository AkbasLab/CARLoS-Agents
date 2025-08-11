from torch import Tensor
import torch
from src.agent import Agent
from src.point import Point
from src.vehicle import Vehicle
from src.environment import Environment
import random
import numpy as np

class Simulation:
    def __init__(
        self, vehicle: Vehicle, environment: Environment, agent: Agent, dt: float = 0.1
    ):
        self.vehicle = vehicle
        self.environment = environment
        self.agent = agent
        self.dt = dt
        self.reset_sim_status()

    def reset_sim_status(self) -> None:
        self.total_time_steps = 0
        self.vehicle_in_lane = True
        self.vehicle_in_motion = True

    def sim_reset(
        self, longitude: float, latitude: float, dir_angle_offset: float, speed: float
    ) -> None:
        self.reset_sim_status()
        center_point, heading = self.environment.position_from_coordinates(
            longitude=longitude,
            latitude=latitude,
            angle_offset=dir_angle_offset,
        )
        self.vehicle.vehicle_setup(center_point, heading, speed)
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

    def sim_random_reset(self, speed_range: list[float] = [10.0, 75.0]):
        longitude = random.uniform(0, 1)
        latitude = random.uniform(0.25, 0.75)
        dir_angle_offset = random.uniform(-np.pi / 4, np.pi / 4)
        speed = random.uniform(speed_range[0], speed_range[1])
        center_point, heading = self.environment.position_from_coordinates(
            longitude=longitude,
            latitude=latitude,
            angle_offset=dir_angle_offset,
        )
        self.vehicle.vehicle_setup(
            center_point=center_point, heading=heading, speed_mph=speed
        )
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

    def get_state(self) -> Tensor:
        _, sensor_data = self.agent.sensors.sense(self.environment, self.vehicle)
        return torch.tensor(
            [self.vehicle.speed_mph, self.vehicle.heading, *sensor_data],
            dtype=torch.float32,
        )

    def sim_step(self) -> None:
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

        state = self.get_state()

        action = self.agent.decide(state)
        steering, acceleration = action[0], action[1]

        self.vehicle.update_position(steering, acceleration, self.dt)

        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

        self.update_sim_status()

        reward = self.agent.compute_reward(
            state, in_lane=self.vehicle_in_lane, in_motion=self.vehicle_in_motion
        )
        return reward

    def update_sim_status(self) -> None:
        self.total_time_steps += 1
        self.vehicle_in_lane = self.environment.point_in_lane(self.vehicle.center_point)
        self.vehicle_in_motion = self.vehicle.speed_mph > 0

    def get_sim_status(self) -> tuple[float, bool, bool]:
        return self.total_time_steps, self.vehicle_in_lane, self.vehicle_in_motion
