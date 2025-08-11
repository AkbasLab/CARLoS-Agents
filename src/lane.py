from src.point import Point
from scipy.interpolate import CubicSpline
import numpy as np

class Lane:
    def __init__(
        self,
        control_points: list[Point],
        lane_width: float = 12.0,
        closed_loop: bool = False,  
    ):
        self.lane_width = lane_width
        self.closed_loop = closed_loop
        self.lane_setup(control_points, lane_width, closed_loop)

    def lane_setup(
        self,
        control_points: list[Point],
        lane_width: float = None,
        closed_loop: bool = None,
    ):
        self.control_points = control_points
        if lane_width is not None:
            self.lane_width = lane_width
        if closed_loop is not None:
            self.closed_loop = closed_loop

        self.center_line, x_center, y_center = self.calculate_center(
            self.control_points
        )
        self.left_edge, self.right_edge = self.calculate_edges(x_center, y_center)

    def calculate_center(
        self, control_points: list[Point]
    ) -> tuple[list[Point], list[float], list[float]]:
        num_points = 500  
        self.x = np.array([p.x for p in control_points])
        self.y = np.array([p.y for p in control_points])

        self.spline_x, self.spline_y, self.t = self.__calculate_xy_spline(
            self.x, self.y
        )

        self.t_center = np.linspace(self.t.min(), self.t.max(), num_points)
        x_center = self.spline_x(self.t_center)
        y_center = self.spline_y(self.t_center)
        center = [Point(x, y) for x, y in zip(x_center, y_center)]

        if self.closed_loop:
            center.append(
                center[0]
            )  
        return center, x_center, y_center

    def __calculate_xy_spline(self, x_pts: list[float], y_pts: list[float]):
        distances = np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)
        t = np.concatenate(
            ([0], np.cumsum(distances))
        )

        if self.closed_loop:
            x_pts = np.append(x_pts, x_pts[0])
            y_pts = np.append(y_pts, y_pts[0])
            t = np.append(t, t[-1] + distances[0]) 

        bc_type = "periodic" if self.closed_loop else "not-a-knot"
        spline_x = CubicSpline(t, x_pts, bc_type=bc_type)
        spline_y = CubicSpline(t, y_pts, bc_type=bc_type)

        return spline_x, spline_y, t

    def __calculate_slope_vectors(self):
        dx_dt = self.spline_x.derivative()(self.t_center)
        dy_dt = self.spline_y.derivative()(self.t_center)

        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        scale_factor = (self.lane_width / 2) / magnitude

        slope_vector_x = -dy_dt * scale_factor  
        slope_vector_y = dx_dt * scale_factor 

        return slope_vector_x, slope_vector_y

    def calculate_edges(self, x_center, y_center) -> tuple[list[Point], list[Point]]:
        slope_vector_x, slope_vector_y = self.__calculate_slope_vectors()

        right_edge = self.calculate_edge_coordinates(
            center_xy=(x_center, y_center),
            slope_vector_xy=(slope_vector_x, slope_vector_y),
            multiplier=-1,
        )

        left_edge = self.calculate_edge_coordinates(
            center_xy=(x_center, y_center),
            slope_vector_xy=(slope_vector_x, slope_vector_y),
            multiplier=1,
        )

        if self.closed_loop:
            right_edge = np.vstack((right_edge, right_edge[0]))
            left_edge = np.vstack((left_edge, left_edge[0]))

        left_edge = [Point(x, y) for x, y in left_edge]
        right_edge = [Point(x, y) for x, y in right_edge]
        return left_edge, right_edge

    def calculate_edge_coordinates(
        self, center_xy, slope_vector_xy, multiplier
    ) -> tuple[float, float]:
        x = center_xy[0] + slope_vector_xy[0] * multiplier
        y = center_xy[1] + slope_vector_xy[1] * multiplier
        edge = np.array([x, y]).T
        return edge 
    
