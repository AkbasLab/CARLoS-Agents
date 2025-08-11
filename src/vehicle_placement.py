import numpy as np
from src.point import Point
from src.lane import Lane

def lateral_adjustment(latitude: float, angle_offset: float) -> float:
    if latitude == 0 and (0 < angle_offset < np.pi):
        angle_offset *= -1
    elif latitude == 1 and (0 > angle_offset > -np.pi):
        angle_offset *= -1
    return angle_offset

def open_loop_adjustment(
    longitude: float, latitude: float, angle_offset: float
) -> float:
    if (longitude == 0 or longitude == 1) and latitude == 0:
        if angle_offset > 0:
            angle_offset = 0
        elif angle_offset < (-np.pi / 2):
            angle_offset = -np.pi / 2
    elif (longitude == 0 or longitude == 1) and latitude == 1:
        if angle_offset < 0:
            angle_offset = 0
        elif angle_offset > (np.pi / 2):
            angle_offset = np.pi / 2
    elif longitude == 0 or longitude == 1:
        if np.pi / 2 < angle_offset < np.pi:
            angle_offset = np.pi / 2
        if -np.pi / 2 > angle_offset > -np.pi:
            angle_offset = -np.pi / 2
    return angle_offset

def interpolate_points(points: list[Point], t: float) -> Point:
    if t <= 0:
        return points[0]
    if t >= 1:
        return points[-1]

    total_dist = sum(
        np.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y)
        for i in range(len(points) - 1)
    )

    target_dist = t * total_dist
    acc_dist = 0.0

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        seg_len = np.hypot(p2.x - p1.x, p2.y - p1.y)
        if acc_dist + seg_len >= target_dist:
            local_t = (target_dist - acc_dist) / seg_len
            x = p1.x + local_t * (p2.x - p1.x)
            y = p1.y + local_t * (p2.y - p1.y)
            return Point(x, y)
        acc_dist += seg_len

    return points[-1]

def get_direction(points: list[Point], t: float, delta: float = 0.01) -> Point:
    t1 = max(0.0, t - delta)
    t2 = min(1.0, t + delta)
    p1 = interpolate_points(points, t1)
    p2 = interpolate_points(points, t2)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    norm = np.hypot(dx, dy)
    return Point(dx / norm, dy / norm) if norm != 0 else Point(0.0, 0.0)

def get_center_point(lane: Lane, longitude: float, latitude: float) -> Point:
    left_pt = interpolate_points(lane.left_edge, longitude)
    right_pt = interpolate_points(lane.right_edge, longitude)

    x = left_pt.x + (right_pt.x - left_pt.x) * latitude
    y = left_pt.y + (right_pt.y - left_pt.y) * latitude
    center_pt = Point(x, y)

    return center_pt
