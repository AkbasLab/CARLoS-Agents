import torch
from src.lane import Lane
from src.point import Point
import numpy as np
import src.vehicle_placement as VP

class Environment:
    def __init__(self, lane: Lane):
        self.lane = lane

    def set_lane(self, lane: Lane):
        self.lane = lane

    def point_in_lane(self, point: Point) -> bool:
        center_dist, _, _ = self.point_position_in_lane(point)
        return center_dist >= 0

    def point_position_in_lane(self, point: Point) -> tuple[float, float, float]:
        center_distance, nearest_index = calc_distance(point, self.lane.center_line)
        classification = abs(center_distance)
        if (
            nearest_index == len(self.lane.center_line) - 1
            and not self.lane.closed_loop
        ):
            in_lane = False
        else:
            in_lane = classification <= (self.lane.lane_width / 2)

        left_dist = abs(calc_distance(point, self.lane.left_edge)[0])
        right_dist = abs(calc_distance(point, self.lane.right_edge)[0])

        if right_dist > left_dist:
            right_dist *= -1
        elif left_dist > right_dist:
            left_dist *= -1

        if not in_lane:
            classification *= -1

        return classification, left_dist, right_dist

    def position_from_coordinates(
        self,
        longitude: float,
        latitude: float,
        angle_offset: float,
    ) -> tuple[Point, float]:
        longitude = np.clip(longitude, 0.0, 1.0)
        latitude = np.clip(latitude, 0.0, 1.0)
        center_point = VP.get_center_point(
            lane=self.lane, longitude=longitude, latitude=latitude
        )

        angle_offset = VP.lateral_adjustment(latitude, angle_offset)
        if not self.lane.closed_loop:
            angle_offset = VP.open_loop_adjustment(longitude, latitude, angle_offset)

        lane_direction = VP.get_direction(self.lane.center_line, longitude)

        heading = np.arctan2(lane_direction.y, lane_direction.x) + angle_offset

        return center_point, heading

def determine_distance(point: Point, segment_pt1: Point, segment_pt2: Point):
    x1, y1 = segment_pt1.values()
    x2, y2 = segment_pt2.values()
    x3, y3 = point.values()

    dx = x2 - x1
    dy = y2 - y1
    segment_len_sq = dx * dx + dy * dy

    if segment_len_sq == 0:
        print("SEGMENT = 0")
        return np.hypot(x3 - x1, y3 - y1)

    t = ((x3 - x1) * dx + (y3 - y1) * dy) / segment_len_sq
    t = max(0.0, min(1.0, t)) 

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.hypot(x3 - closest_x, y3 - closest_y)

def closest_points(point: Point, curve_pts: list[Point]):
    px, py = point.values()
    curve_x = np.array([pt.x for pt in curve_pts])
    curve_y = np.array([pt.y for pt in curve_pts])

    distances = np.sqrt((curve_x - px) ** 2 + (curve_y - py) ** 2)
    nearest_idx = np.argmin(distances)
    distances[nearest_idx] = np.inf
    sectond_neared_idx = np.argmin(distances)
    pt1 = curve_pts[nearest_idx]
    pt2 = curve_pts[sectond_neared_idx]
    if distances[nearest_idx] == distances[sectond_neared_idx]:
        print("!! DISTANCES EQUAL !!")
        distances[sectond_neared_idx] = np.inf
        sectond_neared_idx = np.argmin(distances)
        pt2 = curve_pts[sectond_neared_idx]

    return [pt1, pt2], nearest_idx

def calc_distance(point: Point, curve_pts: list[Point]):
    closest_pts, nearest_idx = closest_points(point, curve_pts)
    distance = determine_distance(point, closest_pts[0], closest_pts[1])

    return round(distance, 5), nearest_idx
