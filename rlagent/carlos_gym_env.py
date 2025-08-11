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

LAYOUT_FILE_PATH = r"C:\Users\hp\OneDrive\Desktop\Working_Carlos\src\layouts\path1.txt"

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
        self.vehicle = Vehicle()

        NUM_SENSORS = 8                   
        SENSOR_LENGTH = 100.0                 
        SENSOR_ANGLE_SPREAD = np.pi / 2    

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
            shape=(self.num_sensors,),
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
        info = {
        'in_lane': in_lane,
        'in_motion': in_motion,
        'position': (self.vehicle.center_point.x, self.vehicle.center_point.y)
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
