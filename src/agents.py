from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(ActorCritic, self).__init__()
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
