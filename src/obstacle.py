class Obstacle:
    def __init__(self, position, radius):
        """
        position: tuple(float, float) - (x, y) coordinates
        radius: float - radius or size of the circular obstacle
        """
        self.position = position
        self.radius = radius