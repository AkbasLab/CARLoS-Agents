import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.layout_utils import load_lane_from_file
from rlagent.simulationrl import SimulationRL
from src.vehicle import Vehicle
from src.lane import Lane
from src.environment import Environment
from src.agent import Agent
from src.sensor_array import SensorArray
from src.presentation_agent import PresentationAgent
from src.graphics import render_simulation
from src.obstacle import Obstacle
from src.point import Point
import matplotlib.pyplot as plt

# ── Bug fix: hardcoded Windows path replaced with a portable relative path.
# Path to the road layout file.
# carlos_gym_env.py lives in rlagent/rlagent/, so one level up reaches the
# project root (rlagent/), and then src/layouts/train_path.txt from there.
# os.path.normpath removes the '..' so the path is unambiguous on all OS.
LAYOUT_FILE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # rlagent/rlagent/
    '..', 'src', 'layouts', 'train_path.txt'     # -> rlagent/src/layouts/train_path.txt
))


def place_obstacle_along_lane(environment, lane, longitudinal_ratio,
                               lateral_offset, radius):
    center_line_pts = lane.center_line
    n_points        = len(center_line_pts)
    idx             = int(longitudinal_ratio * (n_points - 1))

    center_point = center_line_pts[idx]
    if idx < n_points - 1:
        next_point = center_line_pts[idx + 1]
    else:
        next_point = center_line_pts[idx - 1]

    dx     = next_point.x - center_point.x
    dy     = next_point.y - center_point.y
    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        perp_x, perp_y = 0, 0
    else:
        perp_x = -dy / length
        perp_y =  dx / length

    obs_x = center_point.x + lateral_offset * perp_x
    obs_y = center_point.y + lateral_offset * perp_y

    environment.add_obstacle(Obstacle(position=(obs_x, obs_y), radius=radius))


class CarlosGymEnv(gym.Env):
    """
    Gymnasium wrapper around the CARLoS simulator for PPO training and
    safety evaluation.

    Parameters
    ----------
    seed : int | None
        RNG seed for obstacle placement.  None = non-deterministic.
    lane_width_ft : float | None
        Lane width in feet.  None -> baseline 12 ft.
    num_obstacles : int | None
        Number of obstacles to place.  None -> baseline 6.
    reward_weights : dict | None
        Reward weights used during training and evaluation.
        Keys: 'in_lane', 'in_motion', 'collision', 'out_lane'.
        None -> _DEFAULT_REWARD_WEIGHTS below.

    Reward-weight injection
    -----------------------
    SB3's vectorised environments call env.step(action) with a single
    argument - there is no way to pass reward_weights through the SB3
    training loop as a step-level kwarg.  Instead, reward_weights is stored
    as self._reward_weights at __init__ time and read automatically inside
    step().  This means:

      During training  : pass reward_weights to __init__; step() uses them
                         automatically via self._reward_weights.
      During evaluation: pass reward_weights directly to step() as before
                         (explicit argument takes priority over the instance
                         attribute, so evaluation behaviour is unchanged).
    """

    # Default baseline reward weights (paper Section III-C)
    _DEFAULT_REWARD_WEIGHTS = {
        'in_lane'   :  1.0,
        'in_motion' :  0.5,
        'collision' : -10.0,
        'out_lane'  :  -5.0,
    }

    def __init__(self, seed=None, lane_width_ft=None, num_obstacles=None,
                 reward_weights=None):
        super(CarlosGymEnv, self).__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        lane_ctrl_points, lane_width, closed_loop = load_lane_from_file(
            LAYOUT_FILE_PATH
        )

        active_lane_width = lane_width_ft if lane_width_ft is not None else 12
        active_num_obs    = num_obstacles  if num_obstacles  is not None else 6

        self.lane_width_ft = active_lane_width

        # Store reward weights so step() can use them when called by SB3
        # (which only passes the action, not extra kwargs).
        self._reward_weights = (
            reward_weights
            if reward_weights is not None
            else self._DEFAULT_REWARD_WEIGHTS
        )

        self.lane = Lane(
            control_points = lane_ctrl_points,
            lane_width     = active_lane_width,
            closed_loop    = closed_loop,
        )

        self.environment = Environment(self.lane)
        lat_offsets      = [-1.5, -1, -0.5, 0.5, 1, 1.5]
        radii            = [0.5, 1, 1.5, 2]
        # Bug fix: linspace(0, 1, n) places obstacle at longitudinal_ratio=0.0,
        # which is the vehicle spawn point (SEGMENT=0).  The agent collides
        # immediately at reset regardless of its action, causing all episodes to
        # terminate in 1-5 steps.  Restricting to [0.15, 0.85] keeps obstacles
        # away from both the spawn point and the lane end.
        lon_dist         = np.linspace(0.15, 0.85, active_num_obs)

        for num in range(active_num_obs):
            place_obstacle_along_lane(
                self.environment, self.lane,
                longitudinal_ratio = lon_dist[num],
                lateral_offset     = random.choice(lat_offsets),
                radius             = random.choice(radii),
            )

        self.vehicle        = Vehicle()
        NUM_SENSORS         = 12
        SENSOR_LENGTH       = 120.0
        SENSOR_ANGLE_SPREAD = 2 * np.pi

        self.sensor_array = SensorArray(
            num_sensors         = NUM_SENSORS,
            sensor_length       = SENSOR_LENGTH,
            sensor_angle_spread = SENSOR_ANGLE_SPREAD,
        )

        self.agent = PresentationAgent(self.sensor_array)
        self.sim   = SimulationRL(
            vehicle     = self.vehicle,
            environment = self.environment,
            agent       = self.agent,
        )
        self.num_sensors = (
            self.sim.num_sensors if hasattr(self.sim, 'num_sensors') else 12
        )

        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low =np.array([-1.0, 0.0]),
            high=np.array([ 1.0, 1.0]),
            dtype=np.float32,
        )

        self.max_steps    = 200
        self.current_step = 0

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_step = 0
        self.sim.sim_random_reset()
        obs = self._get_observation()
        return obs, {}

    # ─────────────────────────────────────────────────────────────────────
    def calculate_safety_margin(self, vehicle_pos, in_lane=True):
        """
        Signed distance to the nearest lane edge.
          +  inside the lane  (normal operation)
          0  on the boundary
          -  outside the lane (post-failure)
        """
        def dist(p1, p2):
            return ((p1[0] - p2.x) ** 2 + (p1[1] - p2.y) ** 2) ** 0.5

        left_distances  = [dist(vehicle_pos, pt) for pt in self.lane.left_edge]
        right_distances = [dist(vehicle_pos, pt) for pt in self.lane.right_edge]

        unsigned_margin = min(min(left_distances), min(right_distances))
        sign            = 1.0 if in_lane else -1.0
        return sign * unsigned_margin

    # ─────────────────────────────────────────────────────────────────────
    def step(self, action, reward_weights=None):
        """
        Apply action, advance the simulator, return (obs, reward, term, trunc, info).

        reward_weights resolution order
        --------------------------------
        1. Argument passed directly to step()   - used during evaluation.
        2. self._reward_weights set at __init__ - used during SB3 training
           (SB3 calls env.step(action) with no extra kwargs).

        Bug fix: observation ordering
        --------------------------------
        The original code called _get_observation() BEFORE sim_step_rl(),
        so the observation returned to the agent described the state BEFORE
        the action was applied. This creates a one-step lag that corrupts
        the (s, a, r, s') tuples used for policy updates. Fixed order:
          1. sim_step_rl()       - apply action, advance physics
          2. _get_observation()  - read new state AFTER the action
        """
        steering_norm         = float(action[0].item())
        throttle_norm         = float(action[1])
        max_steering_rad      = np.pi / 4
        max_acceleration_mph2 = 20.0

        steering_rad      = steering_norm  * max_steering_rad
        acceleration_mph2 = throttle_norm  * max_acceleration_mph2

        self.current_step += 1

        # Resolve which reward weights to use
        active_rw = reward_weights if reward_weights is not None else self._reward_weights

        # CORRECT ORDER: step simulator FIRST, then read the resulting observation
        reward = self.sim.sim_step_rl(
            external_action = (steering_rad, acceleration_mph2),
            reward_weights  = active_rw,
        )
        obs = self._get_observation()   # state AFTER the action was applied

        # Read simulation status BEFORE deciding termination.
        # We deliberately do NOT use self.sim.is_done() for episode termination.
        # is_done() is a CARLoS internal that returns True when the vehicle
        # crosses back to SEGMENT=0 (the track start line) — on a closed-loop
        # track this fires immediately when the agent spins or reverses,
        # causing all episodes to terminate in 5 steps regardless of the policy.
        # Instead we define termination ourselves from observable events.
        _, in_lane, in_motion = self.sim.get_sim_status()
        collision   = getattr(self.sim, '_collision', False)
        vehicle_pos = (self.vehicle.center_point.x, self.vehicle.center_point.y)

        safety_margin = self.calculate_safety_margin(vehicle_pos, in_lane=in_lane)

        # Termination conditions (our own, not sim.is_done()):
        #   terminated = True  when a safety-critical event ends the episode early
        #   truncated  = True  when the episode reaches max_steps (success)
        collision_done    = collision
        lane_violation_done = not in_lane
        stopped_done      = not in_motion

        terminated = collision_done or lane_violation_done or stopped_done
        truncated  = (self.current_step >= self.max_steps) and not terminated

        # Priority: truncated checked FIRST (simultaneous term+trunc -> success)
        if truncated:
            terminal_event = 'truncated'
        elif collision:
            terminal_event = 'collision'
        elif not in_lane:
            terminal_event = 'lane_violation'
        elif not in_motion:
            terminal_event = 'stopped'
        else:
            terminal_event = None

        info = {
            'in_lane'       : in_lane,
            'in_motion'     : in_motion,
            'collision'     : collision,
            'position'      : vehicle_pos,
            'safety_margin' : safety_margin,
            'terminal_event': terminal_event,
        }
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        _, detection_distances = self.agent.sensors.sense(
            self.environment, self.vehicle
        )
        return np.array(detection_distances, dtype=np.float32)

    def render(self, mode='human'):
        render_simulation(self.sim)
        plt.title('CARLOS RL AGENT SIM')
        plt.pause(0.001)

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass