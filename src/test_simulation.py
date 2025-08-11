from src.simulation import Simulation
from src.vehicle import Vehicle
from src.lane import Lane
from src.agent import SimpleAgent
from src.environment import Environment
from src.sensor_array import SensorArray
import src.layout_utils
import math
import src.graphics
import matplotlib.pyplot as plt
from src.point import Point
import numpy as np

def init_sim():
    lane_ctrl_points = [Point(100, 100), Point(300, 100)]
    lane = Lane(control_points=lane_ctrl_points, lane_width=12.0, closed_loop=False)
    env = Environment(lane)
    v = Vehicle()
    sensor_arr = SensorArray(
        num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi
    )
    agent = SimpleAgent(sensor_array=sensor_arr)
    return Simulation(vehicle=v, environment=env, agent=agent)

def test_sim_reset_left_edge():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(
        longitude=longitude,
        latitude=latitude,
        dir_angle_offset=dir_angle_offset,
        speed=speed,
    )

    plt.subplot(2, 4, 1)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Reset Left Edge - Initial")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    sim.sim_reset(longitude=0.5, latitude=0.0, dir_angle_offset=math.pi / 4, speed=45.0)

    plt.subplot(2, 4, 5)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Vehicle .5 down lane, left edge, 45 deg angle towards center")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    assert (
        199.999999 <= sim.vehicle.center_point.x <= 200.000001
    ), f"Vehicle center point X does not match expected value."
    assert (
        105.000001 <= sim.vehicle.center_point.y <= 106.000002
    ), f"Vehicle center point Y does not match expected value."
    assert (
        ((-math.pi / 4) - 0.0000001)
        <= sim.vehicle.heading
        <= ((-math.pi / 4) + 0.0000001)
    ), f"Heading angle does not match expected value."

    print("Simulation Test: Visualization of left edge reset PASSED. CONFIRM VISUALLY")

def test_sim_reset_right_edge():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(
        longitude=longitude,
        latitude=latitude,
        dir_angle_offset=dir_angle_offset,
        speed=speed,
    )
    plt.subplot(2, 4, 2)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Reset Right Edge - Initial")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    sim.sim_reset(
        longitude=0.5, latitude=1.0, dir_angle_offset=-math.pi / 4, speed=45.0
    )
    plt.subplot(2, 4, 6)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("0.5 down lane, right edge, 45 deg angle towards center")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    assert (
        199.999999 <= sim.vehicle.center_point.x <= 200.000001
    ), f"Vehicle center point X does not match expected value."
    assert (
        93.999999 <= sim.vehicle.center_point.y <= 94.000001
    ), f"Vehicle center point Y does not match expected value."
    assert (
        (math.pi / 4 - 0.0000001) <= sim.vehicle.heading <= (math.pi / 4 + 0.0000001)
    ), f"Heading angle does not match expected value."

    print(
        "Simulation Test: Visualization of right edge reset PASSED. CONFIRM VISUALLY."
    )

def test_sim_random_reset():
    sim = init_sim()
    longitude = 0.0
    latitude = 0.5
    speed = 25.0
    dir_angle_offset = 0.0
    sim.sim_reset(
        longitude=longitude,
        latitude=latitude,
        dir_angle_offset=dir_angle_offset,
        speed=speed,
    )
    plt.subplot(2, 4, 3)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Random Reset - Initial")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    sim.sim_random_reset()
    plt.subplot(2, 4, 7)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Random Reset - After")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    print("Simulation Test: Random reset visualization NEEDS VISUAL CONFIRMATION.")

def test_sim_step():
    sim = init_sim()
    sim.sim_reset(longitude=0.5, latitude=0.5, dir_angle_offset=0.0, speed=25.0)
    original_x = sim.vehicle.center_point.x
    original_y = sim.vehicle.center_point.y
    plt.subplot(2, 4, 4)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Step - Initial")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    sim.sim_step()
    plt.subplot(2, 4, 8)
    src.graphics.render_simulation_subplots(sim=sim)
    plt.title("Simulation Test - Step - After")
    plt.xlim(50, 350)
    plt.ylim(50, 150)

    assert sim.total_time_steps == 1, f"Total time steps should be 1 after one step."
    assert (
        sim.vehicle.center_point.x != original_x
    ), f"Vehicle should have moved after one step."
    assert (
        sim.vehicle.center_point.y != original_y
    ), f"Vehicle should have moved after one step."

    print("Simulation Test: Simulation step PASSED. CONFIRM VISUALLY")

def test_sim_status():
    sim = init_sim()
    sim.sim_random_reset()
    sim_status = sim.get_sim_status()
    assert sim_status[0] == 0, f"Total time steps should be 0 after initialization."
    assert sim_status[1] == True, f"Vehicle should be in lane after initialization."
    assert sim_status[2] == True, f"Vehicle should be in motion after initialization."
    sim.sim_step()
    sim_status = sim.get_sim_status()
    assert sim_status[0] == 1, f"Total time steps should be 1 after one step."
    print("Simulation Test: Sim status update PASSED.")

def test_get_state():
    sim = init_sim()
    sim.sim_random_reset()
    state = np.array(sim.get_state())
    assert (
        len(state) == 7
    ), f"State should contain 7 elements: heading angle, speed, and 5 detection distances."

    print("Simulation Test: State retrieval PASSED.")

def run_tests():
    test_sim_reset_left_edge()
    test_sim_reset_right_edge()
    test_sim_random_reset()
    test_sim_status()
    test_sim_step()
    test_get_state()
    src.graphics.show_without_pause(
        title="Simulation Test - Step - After", x_lim=[50, 350], y_lim=[50, 150]
    )
    print("Simulation Test: All tests PASSED.\n")

if __name__ == "__main__":
    run_tests()