from src.point import Point
from src.lane import Lane
from tkinter import filedialog
from src.vehicle import Vehicle
from src.sensor_array import SensorArray

def load_lane_from_file(file_path: str = None) -> tuple[list[Point], float, bool]:
    if file_path is None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if file_path is None or len(file_path) == 0:
        return None

    with open(file_path, "r") as file:
        lines = file.readlines()
        control_points = []
        lane_width = 0.0
        closed_loop = False
        for line in lines:
            if line.startswith("L:"):
                closed_loop = line[2:].strip().lower() == "true"
            elif line.startswith("W:"):
                lane_width = float(line[2:].strip())
            elif line.startswith("P:"):
                point_data = line[2:].strip().split(",")
                point = Point(float(point_data[0]), float(point_data[1]))
                control_points.append(point)

    return control_points, lane_width, closed_loop

def save_lane_to_file(filename: str, lane: Lane) -> None:
    save_layout_to_file(
        filename=filename,
        ctrl_pts=lane.control_points,
        closed_loop=lane.closed_loop,
        lane_width=lane.width,
    )

def save_layout_to_file(
    filename: str, ctrl_pts: list, closed_loop: bool, lane_width: float
) -> None:
    if filename is None:
        filename = filedialog.asksaveasfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if filename is None or len(filename) == 0:
        return None

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    with open(filename, "w") as file:
        file.write(f"L:{closed_loop}\n")
        file.write(f"W:{lane_width}\n")
        for point in ctrl_pts:
            file.write(f"P:{point.x},{point.y}\n")

def save_vehicle_setup(filename: str, save_data: str):
    if filename is None:
        filename = filedialog.asksaveasfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if filename is None or len(filename) == 0:
        return None

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    with open(filename, "w") as file:
        for k, v in save_data.items():
            file.write(f"{k}:{v}\n")

def load_vehicle_setup_from_file(file_path: str = None) -> dict:
    if file_path is None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if file_path is None or len(file_path) == 0:
        return None

    with open(file_path, "r") as file:
        lines = file.readlines()
        presets = {}
        for line in lines:
            data = line.split(":")
            if len(data) != 2:
                continue
            key = data[0].strip()
            value = data[1].strip()
            presets[key] = value

    return presets
