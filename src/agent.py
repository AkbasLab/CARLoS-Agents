from torch import Tensor
from src.sensor_array import SensorArray
from src.environment import Environment
from src.vehicle import Vehicle
import torch
import os
import src.carlos_logging

class Agent:
    def __init__(self, sensor_array: SensorArray):
        self.sensors = sensor_array

    def decide(self, state):
        raise NotImplementedError("decide() method not implemented in child class.")

    def compute_reward(self, state: Tensor, in_lane: bool, in_motion: bool) -> float:
        raise NotImplementedError(
            "compute_reward() method not implemented in child class."
        )
    def sense(self, env: Environment, vehicle: Vehicle):
        sensor_data = self.sensor_array.sense(env, vehicle)
        return sensor_data

    def save(self, dir_path="./checkpoints", tag="latest"):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.model.actor.state_dict(),
                "critic_state_dict": self.model.critic.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            os.path.join(dir_path, f"agent_{tag}.pt"),
        )
        src.carlos_logging.log_message(
            f"Saved model checkpoint to {dir_path}/agent_{tag}.pt"
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.model.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        src.carlos_logging.log_message(f"Loaded model checkpoint from {path}")

class SimpleAgent(Agent):
    def __init__(self, sensor_array: SensorArray):
        super().__init__(sensor_array)

    def decide(self, state):
        return 1.0, 1.0

    def compute_reward(self, state, in_lane: bool, in_motion: bool) -> float:
        return 1.0 if in_lane and in_motion else -1.0
