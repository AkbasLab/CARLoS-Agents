import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.layout_utils import load_lane_from_file
from rlagent.simulationrl import SimulationRL
from src.vehicle import Vehicle
from src.lane import Lane
from src.environment import Environment
from src.agent import Agent
from src.sensor_array import SensorArray
from src.presentation_agent import PresentationAgent 
from src.graphics import render_simulation
import matplotlib.pyplot as plt
from src.obstacle import Obstacle
from src.point import Point

LAYOUT_FILE_PATH = r".\src\layouts\train_path.txt"

from src.point import Point

def place_obstacle_along_lane(environment, lane, longitudinal_ratio, lateral_offset, radius):
    center_line_pts = lane.center_line
    n_points = len(center_line_pts)
    idx = int(longitudinal_ratio * (n_points - 1))
    
    center_point = center_line_pts[idx]
    if idx < n_points - 1:
        next_point = center_line_pts[idx + 1]
    else:
        next_point = center_line_pts[idx - 1]
    
    dx = next_point.x - center_point.x
    dy = next_point.y - center_point.y
    length = (dx ** 2 + dy ** 2) ** 0.5
    
    perp_x = -dy / length
    perp_y = dx / length
    
    obs_x = center_point.x + lateral_offset * perp_x
    obs_y = center_point.y + lateral_offset * perp_y
    
    environment.add_obstacle(Obstacle(position=(obs_x, obs_y), radius=radius))
    print(f"Placed obstacle at: {(obs_x, obs_y)}, radius: {radius}")

class CarlosGymEnv(gym.Env):
    def __init__(self):
        super(CarlosGymEnv, self).__init__()
        lane_ctrl_points, lane_width, closed_loop = load_lane_from_file(LAYOUT_FILE_PATH)
        self.lane = Lane(
            control_points=lane_ctrl_points,
            lane_width=lane_width,
            closed_loop=closed_loop
        )
        self.environment = Environment(self.lane)
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.16, lateral_offset=1.5, radius=0.5)
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.32, lateral_offset=-1.0, radius=1)
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.48, lateral_offset=-0.5, radius=2.5)
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.64, lateral_offset=-1.0, radius=0.5) 
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.80, lateral_offset=1.5, radius=1)
        place_obstacle_along_lane(self.environment, self.lane, longitudinal_ratio=0.96, lateral_offset=-1.0, radius=2.5)

        self.vehicle = Vehicle()

        NUM_SENSORS = 12                   
        SENSOR_LENGTH = 120.0                 
        SENSOR_ANGLE_SPREAD = 2 * np.pi    

        self.sensor_array = SensorArray(
            num_sensors=NUM_SENSORS,
            sensor_length=SENSOR_LENGTH,
            sensor_angle_spread=SENSOR_ANGLE_SPREAD,
        )   

        self.agent = PresentationAgent(self.sensor_array)
        self.sim = SimulationRL(vehicle=self.vehicle, environment=self.environment, agent=self.agent)
        self.num_sensors = self.sim.num_sensors if hasattr(self.sim, 'num_sensors') else 10

        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(12,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.sim.sim_random_reset()
        obs = self._get_observation()
        return obs, {}
    
    def calculate_safety_margin(self, vehicle_pos):
        # Helper to compute shortest distance from vehicle to lane edges
        def dist(p1, p2):
            return ((p1[0] - p2.x) ** 2 + (p1[1] - p2.y) ** 2) ** 0.5

        left_distances = [dist(vehicle_pos, pt) for pt in self.lane.left_edge]
        right_distances = [dist(vehicle_pos, pt) for pt in self.lane.right_edge]

        min_left = min(left_distances)
        min_right = min(right_distances)

        safety_margin = min(min_left, min_right)
        return safety_margin


    def step(self, action):
        steering_norm = float(action[0])
        throttle_norm = float(action[1])

        max_steering_rad = np.pi / 4
        max_acceleration_mph2 = 20.0

        steering_rad = steering_norm * max_steering_rad
        acceleration_mph2 = throttle_norm * max_acceleration_mph2

        self.current_step += 1

        obs = self._get_observation()
        reward = self.sim.sim_step_rl(external_action=(steering_rad, acceleration_mph2))

        terminated = self.sim.is_done()
        truncated = self.current_step >= self.max_steps
        _, in_lane, in_motion = self.sim.get_sim_status()
        collision = getattr(self.sim, '_collision', False)
        vehicle_pos = (self.vehicle.center_point.x, self.vehicle.center_point.y)
        safety_margin = self.calculate_safety_margin(vehicle_pos)
        info = {
        'in_lane': in_lane,
        'in_motion': in_motion,
        'collision': collision,
        'position': (self.vehicle.center_point.x, self.vehicle.center_point.y),
        'safety_margin': safety_margin
    }
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        _, detection_distances = self.agent.sensors.sense(self.environment, self.vehicle)
        return np.array(detection_distances, dtype=np.float32)

    def render(self, mode='human'):
        render_simulation(self.sim)
        plt.title("CARLOS RL AGENT SIM")
        plt.pause(0.001)

    def close(self):
        self.sim.close()
