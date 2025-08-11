import numpy as np
import torch
from src.agent import Agent
from torch import Tensor, nn, optim
from src.sensor_array import SensorArray

class ActorCriticModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
            nn.Tanh(),  
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class NewAgent(Agent):
    def __init__(
        self,
        sensor_array: SensorArray,
        lr=1e-3,
        input_scaler=None,
        max_accel: float = 10.0,
    ):
        self.num_inputs = 2 + sensor_array.num_sensors  
        self.input_scaler = input_scaler or torch.tensor(
            [0.001] * self.num_inputs + [np.pi, 70.0]
        )
        self.max_accel = max_accel
        self.model = ActorCriticModel(self.num_inputs, 2)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

        super().__init__(sensor_array)

    def decide(self, state: Tensor) -> Tensor:
        action, _critic_out = self.model(state.reshape(-1, self.num_inputs))
        self.last_state = state

        return (action * torch.tensor([np.pi, self.max_accel])).flatten()

    def compute_reward(self, state, in_lane, in_motion):
        speed, steering_angle, *sensor_data = state
        reward = 0.0

        reward += 1.0 if in_lane else -2.0

        reward += 0.5 if in_motion else -0.5

        reward += 0.1 * (speed / 75.0)

        min_sensor = min(sensor_data)
        if min_sensor < 1.0:
            reward -= 1.0 - min_sensor

        reward -= 0.01

        return torch.tensor([reward])

    def train_step(self, next_state: Tensor, reward: Tensor, done: bool):
        next_state = next_state.reshape(-1, self.num_inputs)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            target = reward + (
                0.99 * next_value * (1.0 - float(done))
            )  

        action, value = self.model(self.last_state.reshape(-1, self.num_inputs))

        value_loss = nn.functional.mse_loss(value, target)

        log_prob = -(
            (action - action.detach()) ** 2
        ).mean()  
        advantage = (target - value).detach()
        actor_loss = -log_prob * advantage

        loss = value_loss + actor_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
