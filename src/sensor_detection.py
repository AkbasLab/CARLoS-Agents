from src.point import Point
from src.sensor import Sensor
from src.lane import Lane
import numpy as np


def get_lane_detection(sensor: Sensor, lane: Lane) -> float:
    right_edge_point, right_edge_intersection = intersections_on_line_segment(
        lane.right_edge, sensor.origin_point, sensor.end_point
    )
    right = (
        right_edge_intersection
        if right_edge_intersection != -1
        else sensor.sensor_length
    )
    left_edge_point, left_edge_intersection = intersections_on_line_segment(
        lane.left_edge, sensor.origin_point, sensor.end_point
    )
    left = (
        left_edge_intersection if left_edge_intersection != -1 else sensor.sensor_length
    )
    pts = [right_edge_point, left_edge_point]
    intersects = [right, left]
    idx = np.argmin(intersects)

    return pts[idx], intersects[idx]

def intersections_on_line_segment(pt_list: list[Point], pt1: Point, pt2: Point):
    distances = []
    points = []
    for i in range(len(pt_list) - 1):
        intersection = line_segment_intersection(pt1, pt2, pt_list[i], pt_list[i + 1])
        if intersection is not None:
            dist = intersection.distanceTo(pt1)
            distances.append(dist)
            points.append(intersection)

    if len(distances) != 0:
        closest_distance = min(distances)
        closest_point = points[distances.index(closest_distance)]
    else:
        closest_point = None
        closest_distance = -1.0
    return closest_point, closest_distance

def line_segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point):
    r = p2 - p1
    s = q2 - q1
    q_minus_p = q1 - p1
    r_cross_s = cross(r, s)
    qmp_cross_r = cross(q_minus_p, r)

    if r_cross_s == 0:
        return None 
    t = cross(q_minus_p, s) / r_cross_s
    u = qmp_cross_r / r_cross_s

    intersection_point = None

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = p1 + t * r

    return intersection_point

def cross(v1: Point, v2: Point) -> float:
    return v1.x * v2.y - v1.y * v2.x
