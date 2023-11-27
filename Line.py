import numpy as np
import cv2

class Line:
    """
    Represents a 2D line defined by two points (x1, y1) and (x2, y2).
    The line is characterized by its slope and intercept (bias).

    Methods:
    compute_slope: Calculates the slope of the line.
    compute_bias: Calculates the intercept of the line.
    get_coords: Returns the coordinates of the line.
    set_coords: Sets new coordinates for the line.
    draw: Draws the line on an image.
    """

    def __init__(self, x1, y1, x2, y2):
        self.set_coords(x1, y1, x2, y2)
        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        # Avoid division by zero by adding a small epsilon to the denominator
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        # Ensure that coordinates are stored as float32 for consistency
        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

    def draw(self, img, color=[255, 0, 0], thickness=10):
        # Draw the line on the given image
        cv2.line(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)

