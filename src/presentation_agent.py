from src.agent import Agent
from src.sensor_array import SensorArray
import torch

class PresentationAgent(Agent):
    def __init__(self, sensor_array: SensorArray):
        super().__init__(sensor_array)

    def decide(self, state):
        accel = 0.5
        detections = state[2:]
        index = torch.argmax(detections).item()
        steering = self.sensors.sensors[index].angle_offset
        return steering, accel

    def compute_reward(self, state, in_lane: bool, in_motion: bool, collision=False, weights=None) -> float:
        if weights is None:
            weights = {
            'in_lane': 1.0,
            'in_motion': 0.5,
            'collision': -50.0,
            'out_lane': -10.0
            }

        reward = 0.0

        if collision:
            reward += weights['collision']
        else:
            reward += weights['in_lane'] if in_lane else weights['out_lane']
            reward += weights['in_motion'] if in_motion else 0.0

        return reward

