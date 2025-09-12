import matplotlib.pyplot as plt
from src.lane import Lane
from src.simulation import Simulation
from src.vehicle import Vehicle
from src.environment import Environment
from src.point import Point
from src.sensor_array import SensorArray
import numpy as np

def list_points_as_values(point_list: list[Point]) -> tuple[list[float], list[float]]:
    x = []
    y = []
    for p in point_list:
        x.append(p.x)
        y.append(p.y)
    return x, y

def plot_environment(environment: Environment, ax=None):  
    lane = environment.lane
    plot_lane(lane, ax) 

def plot_lane(lane: Lane, ax=None): 
    ax = plt.gca() if ax is None else ax

    center_x, center_y = list_points_as_values(lane.center_line)
    ax.plot(center_x, center_y, "k--", label="Center Line")

    left_x, left_y = list_points_as_values(lane.left_edge)
    ax.plot(left_x, left_y, "b-", label="Left Edge")

    right_x, right_y = list_points_as_values(lane.right_edge)
    ax.plot(right_x, right_y, "m-", label="Right Edge")

    ax.fill(
        np.concatenate((left_x, right_x[::-1])),
        np.concatenate((left_y, right_y[::-1])),
        color="gray",
        alpha=0.5,
        label="Lane Area",
    )

def plot_vehicle(vehicle: Vehicle, ax=None): 
    ax = plt.gca() if ax is None else ax
    body_x, body_y = list_points_as_values(vehicle.body.corners)
    ax.fill(body_x, body_y, "b", label="Vehicle Body")

    ax.plot(
        vehicle.center_point.x, vehicle.center_point.y, "ro", label="Vehicle Center"
    )

    d = np.array(vehicle.get_direction()) * 10.0
    heading_point = vehicle.center_point + Point(d[0], d[1])
    x = [vehicle.center_point.x, (heading_point.x)]
    y = [vehicle.center_point.y, (heading_point.y)]
    ax.plot(x, y, "g-", label="Vehicle Heading")
    ax.annotate(
        "",
        xy=(x[1], y[1]),
        xytext=(x[0], y[0]),
        arrowprops=dict(
            facecolor="green",
            edgecolor="green",
            arrowstyle="->",
            lw=2,
        ),
    )

def plot_sensors(sensor_array: SensorArray, ax=None):  
    ax = plt.gca() if ax is None else ax
    label = "Sensor"
    for i, sensor in enumerate(sensor_array.sensors):
        ax.plot(
            [sensor.origin_point.x, sensor.end_point.x],
            [sensor.origin_point.y, sensor.end_point.y],
            "r--",
            label=label,
        )
        label = None

def plot_sensor_detections(detection_points, detection_distances, ax=None):
    ax = plt.gca() if ax is None else ax
    label = "Sensor Detection"
    for i, point in enumerate(detection_points):
        if point is not None:
            ax.plot(point.x, point.y, "kx", label=label)
            ax.annotate(
                text=f"{detection_distances[i]: .2f}",
                xy=(point.x, point.y),
                fontsize=8,
                ha="right",
            )
            label = None


def show(
    title: str = "", x_lim: list[float] = [0, 400], y_lim: list[float] = [0, 400]
):
    plt.tight_layout()
    plt.legend()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.title(title)
    plt.pause(0.000000001)


def show_without_pause(
    title: str = "", x_lim: list[float] = [0, 400], y_lim: list[float] = [0, 400]
): 
    plt.tight_layout()
    plt.legend()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.title(title)
    plt.show()


def render_simulation(sim: Simulation):
    plt.clf()
    env = sim.environment
    plot_environment(env)  
    for obstacle in env.obstacles:
        circle = plt.Circle(obstacle.position, obstacle.radius, color='red')
        plt.gca().add_patch(circle)

    vehicle = sim.vehicle
    plot_vehicle(vehicle)  
    sensors = sim.agent.sensors  
    points, detections = sensors.sense(env=env, vehicle=vehicle) 
    plot_sensors(sensor_array=sensors)
    plot_sensor_detections(
        detection_points=points, detection_distances=detections
    )

def render_simulation_subplots(sim: Simulation):
    env = sim.environment
    plot_environment(env) 
    vehicle = sim.vehicle
    plot_vehicle(vehicle)  
    sensors = sim.agent.sensors 
    points, detections = sensors.sense(env=env, vehicle=vehicle)  
    plot_sensors(sensor_array=sensors)
    plot_sensor_detections(
        detection_points=points, detection_distances=detections
    )  