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

    def compute_reward(self, state, in_lane: bool, in_motion: bool) -> float:
        return 1.0 if in_lane and in_motion else -1.0
