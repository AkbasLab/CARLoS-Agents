from src.vehicle import Vehicle
from src.lane import Lane
from src.environment import Environment
from src.simulation import Simulation
from src.sensor_array import SensorArray
import math
import src.layout_utils
import src.carlos_logging
import time
import matplotlib.pyplot as plt
import src.graphics
from src.presentation_agent import PresentationAgent
from src.point import Point

def init_log(file_path: str = None):
    curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if file_path is None:
        carlos_logging.init_logger(f"./logs/{curr_time}_carlos_app.log")
    else:
        src.carlos_logging.init_logger(file_path)

init_log()
src.carlos_logging.log_message("Carlos App Initialized")

LAYOUT_FILE_PATH = r"C:\Users\hp\OneDrive\Desktop\Working_Carlos\src\layouts\suvrat_test.txt"
MAX_STEPS = 200
MAX_EPISODES = 100  
NUM_SENSORS = 9
SENSOR_LENGTH = 200.0
SENSOR_ANGLE_SPREAD = math.pi
TIME_STEP_SEC = 0.1  
INITIAL_SPEED_MPH = 10.0  
INITIAL_LONGITUDE = 0.98  
INITIAL_LATITUDE = 0.5  
INITIAL_DIR_ANGLE_OFFSET = 0.0

x_lim = [0, 400]
y_lim = [0, 400]

lane_ctrl_points, lane_width, closed_loop = src.layout_utils.load_lane_from_file(
    LAYOUT_FILE_PATH
)

lane = Lane(
    control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop
)

env = Environment(lane)

vehicle = Vehicle()

sensor_array = SensorArray(
    num_sensors=NUM_SENSORS,
    sensor_length=SENSOR_LENGTH,
    sensor_angle_spread=SENSOR_ANGLE_SPREAD,
)

agent = PresentationAgent(sensor_array)
sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)

src.carlos_logging.log_message("Simulation Initialized")

plt.ion() 
plt.figure(figsize=(8, 8))

def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"

def execute_simulation(sim: Simulation, render: bool = False) -> list[float]:
    total_reward=0.0
    speed=0.0
    done = False
    steps = 0
    while not done and steps < MAX_STEPS:
        reward = sim.sim_step()

        _, in_lane, in_motion = sim.get_sim_status()
        done = not in_lane or not in_motion

        if render:
            src.graphics.render_simulation(
                sim=sim,
            )
            plt.title("CARLOS Execution Example")
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.pause(0.001)
    
        total_reward += reward
        steps += 1

        speed += sim.vehicle.speed_mph

    avg_speed = speed / steps

    return steps, reward, avg_speed

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

while True:
    reward_log = []
    step_count_log = []
    sim.sim_random_reset()
    start_time = time.time()
    total_reward, steps, avg_speed = execute_simulation(
        sim=sim, render=True
    )
    step_count_log.append(steps)
    reward_log.append(total_reward)
    src.carlos_logging.log_message(
        f"[{elapsed_time(start_time)}] | "
        f"Total Reward: {total_reward:.2f} | Steps: {steps} | "
        f"Distance: {round(sim.vehicle.distance_travelled_ft, 3)} ft | Avg. Speed: {round(avg_speed, 2)} mph"
    )
