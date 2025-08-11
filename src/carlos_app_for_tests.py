from src.new_agent import NewAgent
from src.point import Point
from src.vehicle import Vehicle
from src.lane import Lane
from src.environment import Environment
from src.simulation import Simulation
from src.sensor_array import SensorArray
import math
import src.layout_utils
import src.carlos_logging
from src.summer_agent import SummerAgent
import time 
import matplotlib.pyplot as plt
import src.graphics
from src.presentation_agent import PresentationAgent

def init_log(file_path: str = None):
    curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if file_path is None:
        src.carlos_logging.init_logger(f"./logs/{curr_time}_carlos_app.log")
    else:
        src.carlos_logging.init_logger(file_path)

init_log() 
src.carlos_logging.log_message("Carlos App Initialized")
LAYOUT_FILE_PATH = "./layouts/open_loop_0.txt"
MAX_STEPS = 200
MAX_EPISODES = 100 
NUM_SENSORS = 9
SENSOR_LENGTH = 200.0
SENSOR_ANGLE_SPREAD = math.pi
TIME_STEP_SEC = 0.1  
INITIAL_SPEED_MPH = 25.0 
INITIAL_LONGITUDE = 0.98  
INITIAL_LATITUDE = 0.5  
INITIAL_DIR_ANGLE_OFFSET = 0.0  
confirm = False 
x_lim = [0, 300]
y_lim = [0, 300]
lane_ctrl_points = [Point(50, 150), Point(250, 150)]
lane = Lane(control_points=lane_ctrl_points, lane_width=12, closed_loop=False)
env = Environment(lane)

vehicle = Vehicle()

sensor_array = SensorArray(
    num_sensors=NUM_SENSORS,
    sensor_length=SENSOR_LENGTH,
    sensor_angle_spread=SENSOR_ANGLE_SPREAD,
)

ACTION_DIM = 2
MAX_ACCEL = vehicle.max_acceleration_fps2
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
obs_size = NUM_SENSORS + 2  
agent = PresentationAgent(sensor_array)

sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)
sim.sim_reset(
    longitude=INITIAL_LONGITUDE,
    latitude=INITIAL_LATITUDE,
    dir_angle_offset=INITIAL_DIR_ANGLE_OFFSET,
    speed=INITIAL_SPEED_MPH,
)

src.carlos_logging.log_message("Simulation Initialized")

def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"

def execute_simulation(
    sim: Simulation, train: bool = True, render: bool = False
) -> list[float]:
    reward_log = []
    step_count_log = []
    start_time = time.time()
    src.carlos_logging.log_message("Simulation Execution Started")

    for episode in range(MAX_EPISODES):
        sim.sim_random_reset()

        total_reward = 0
        done = False
        steps = 0
        speed = sim.vehicle.speed_mph

        while not done and steps < MAX_STEPS:
            reward = sim.sim_step()

            next_state = sim.get_state()

            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            if train:
                sim.agent.train_step(next_state, reward, done)

            if render:
                src.graphics.render_simulation(
                    sim=sim,
                )
                src.graphics.show(
                    title="CARLOS Execution Example",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )

            state = next_state
            total_reward += reward
            steps += 1

            speed += sim.vehicle.speed_mph

        avg_speed = speed / steps

        step_count_log.append(steps)
        reward_log.append(total_reward)
        src.carlos_logging.log_message(
            f"[{elapsed_time(start_time)}] | Episode {episode+1}/{MAX_EPISODES} | "
            f"Total Reward: {total_reward:.2f} | Steps: {steps} | "
            f"Distance: {round(sim.vehicle.distance_travelled_ft, 3)} ft | Avg. Speed: {round(avg_speed, 2)} mph"
        )
    return reward_log, step_count_log

def plot_rewards(reward_log, window=50):
    avg_rewards = []
    for i in range(len(reward_log)):
        start = max(0, i - window + 1)
        avg = sum(reward_log[start : i + 1]) / (i - start + 1)
        avg_rewards.append(avg)
    plt.figure(figsize=(10, 5))
    plt.plot(reward_log, label="Total Reward per Episode", alpha=0.3)
    plt.plot(avg_rewards, label=f"Moving Avg (window={window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("training")
while True:
    reward_log, step_count_log = execute_simulation(sim=sim, train=False, render=True)

