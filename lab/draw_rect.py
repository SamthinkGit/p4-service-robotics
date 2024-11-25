import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def draw_rectangle(
    dst,
    color: int,
    center: tuple[int, int],
    width,
    height,
    rotation=0,
    filled: bool = False,
):
    points = []

    x, y = center
    radius = np.sqrt((height / 2) ** 2 + (width / 2) ** 2)
    angle = np.arctan2(height / 2, width / 2)
    angles = [angle, -angle + np.pi, angle + np.pi, -angle]
    rot_radians = (np.pi / 180) * -rotation

    for angle in angles:
        y_offset = -1 * radius * np.sin(angle + rot_radians)
        x_offset = radius * np.cos(angle + rot_radians)
        points.append((y + y_offset, x + x_offset))

    points = np.array(points, dtype=np.int32)
    if filled:
        cv2.fillPoly(dst, [points], color)
    else:
        cv2.polylines(dst, [points], isClosed=True, color=color, thickness=1)

    i = points[0][1]
    j = points[0][0]
    r = 3
    dst[i - r : i + r, j - r : j + r] = 50


rotation = 0

while True:

    x = 80
    y = 20

    rotation += 10
    mat = np.zeros((100, 100))
    draw_rectangle(mat, 255, (y, x), 40, 20, rotation, False)

    print(f"Rotation: {rotation}")
    cv2.imshow("Rotated Rectangle", mat)
    cv2.waitKey(500)
