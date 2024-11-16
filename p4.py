import HAL  # type: ignore # noqa
import GUI  # type: ignore # noqa
from enum import Enum
from dataclasses import dataclass, field
from colorama import Fore, Style
import numpy as np


class ColorCode(Enum):
    BLACK = 0
    GRAY = 100
    WHITE = 127
    RED = 128
    ORANGE = 129
    YELLOW = 130
    GREEN = 131
    INDIGO = 133
    VIOLET = 134


OCCUPIED_PIXEL = 0.9
FREE_PIXEL = 0.1
KEYPOINT_SIZE = 2
KEYPOINT_COLOR = ColorCode.YELLOW.value
KEYPOINT_BORDER_COLOR = ColorCode.ORANGE.value

REGISTRY_SCALE_X = 21.5
REGISTRY_SCALE_Y = 20
REGISTRY_ROTATION = np.deg2rad(90)
REGISTRY_TX = 207
REGISTRY_TY = 145
ROTATION_FORCE = 0.2
ROTATION_FORCE_FORWARDING = 0.3
VELOCITY = 0.2

YAW_PRECISSION = 0.2
YAW_PRECISSION_FORWARDING = 0.05
DISTANCE_PRECISSION = 2
VECTOR_PRECISSION = DISTANCE_PRECISSION / 2
ROBOT_YAW_DISPLACEMENT = 90


@dataclass
class Vector:
    module: float
    alpha: float

    @classmethod
    def from_coordinates(cls, src: "Coordinate", dst: "Coordinate") -> "Vector":
        dx = dst.col - src.col
        dy = dst.row - src.row

        # We invert dy since the coordinates uses top as 0 and bottom as max
        # opposed to mathematical 0 at bottom.
        dy = -dy

        module = np.sqrt(dx**2 + dy**2)

        # We return 0 if the module is negligible to avoid the
        # module returning huge values
        if abs(module) < VECTOR_PRECISSION:
            return Vector(0, None)

        alpha = np.arcsin(dy / module)

        return Vector(module, alpha)


@dataclass
class Coordinate:
    row: int
    col: int
    yaw: float = field(default=0.0)

    @classmethod
    def from_world_coordinates(cls, x: float, y: float):
        matrix = np.array(
            [
                [np.cos(REGISTRY_ROTATION), -np.sin(REGISTRY_ROTATION), REGISTRY_TX],
                [np.sin(REGISTRY_ROTATION), np.cos(REGISTRY_ROTATION), REGISTRY_TY],
                [0, 0, 1],
            ]
        )

        vec = np.array([-x * REGISTRY_SCALE_X, y * REGISTRY_SCALE_Y, 1]).transpose()
        result = matrix @ vec
        return cls(row=int(result[1]), col=int(result[0]))


class Map:

    def __init__(self):
        self.map = GUI.getMap("/resources/exercises/amazon_warehouse/images/map.png")

        gui = (
            np.ones((self.map.shape[0], self.map.shape[1]), dtype=int)
            * ColorCode.GRAY.value
        )
        gui[np.all(self.map > [OCCUPIED_PIXEL] * 3, axis=-1)] = ColorCode.WHITE.value
        gui[np.all(self.map < [FREE_PIXEL] * 3, axis=-1)] = ColorCode.BLACK.value

        self.gui = gui

    def show(self):
        GUI.showNumpy(self.gui)

    def keypoint(self, coord: Coordinate):
        self.gui[
            coord.row - KEYPOINT_SIZE : coord.row + KEYPOINT_SIZE,  # noqa
            coord.col - KEYPOINT_SIZE : coord.col + KEYPOINT_SIZE,  # noqa
        ] = KEYPOINT_COLOR
        border_size = KEYPOINT_SIZE + 1
        self.gui[
            coord.row - border_size : coord.row + border_size,  # noqa
            coord.col - border_size : coord.col + border_size,  # noqa
        ] = KEYPOINT_BORDER_COLOR

    def clean(self):
        self.__init__()


class Navigation:

    @property
    def yaw(self) -> float:
        yaw = HAL.getPose3d().yaw

        # We displace the yaw of the robot, since it is shifted
        # pointing up instead of right (mathematical default 0rad)
        return displace_angle(yaw, np.deg2rad(ROBOT_YAW_DISPLACEMENT))

    @property
    def coords(self) -> Coordinate:
        x, y = HAL.getPose3d().x, HAL.getPose3d().y
        result = Coordinate.from_world_coordinates(x, y)
        result.yaw = self.yaw
        return result

    def wait_for_activation(self):
        while self.yaw is None:
            pass
        debug("Navigator is now active")

    def move_to(self, dst: Coordinate):

        yaw_error = float("inf")
        while abs(yaw_error) > YAW_PRECISSION:
            src = self.coords
            target = Vector.from_coordinates(src, dst)
            yaw_error = shortest_angle_distance_radians(src.yaw, target.alpha)

            HAL.setW(yaw_error * ROTATION_FORCE)

        distance = float("inf")
        while abs(distance) > DISTANCE_PRECISSION:
            src = self.coords
            target = Vector.from_coordinates(src, dst)
            distance = target.module
            yaw_error = shortest_angle_distance_radians(src.yaw, target.alpha)

            HAL.setV(VELOCITY)
            if abs(yaw_error) > YAW_PRECISSION_FORWARDING:
                HAL.setW(yaw_error * ROTATION_FORCE_FORWARDING)

        HAL.setV(0)
        yaw_error = float("inf")
        while abs(yaw_error) > YAW_PRECISSION:
            src = self.coords
            yaw_error = shortest_angle_distance_radians(src.yaw, dst.yaw)
            HAL.setW(yaw_error * ROTATION_FORCE)

        HAL.setW(0)


def shortest_angle_distance_radians(a: float, b: float):
    """Calculates the shortest angular distance between two angles in radians.
    This function ensures that the distance is minimized by accounting for the
    circular nature of angular measurements."""
    a = a % (2 * np.pi)
    b = b % (2 * np.pi)

    diff = b - a
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    return diff


def displace_angle(initial, displacement):
    """Adjusts an initial angle by a specified displacement, ensuring
    the result remains within the range of -π to π."""
    result = initial + displacement
    return (result + np.pi) % (2 * np.pi) - np.pi


def warn(text: str) -> None:
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}{Fore.YELLOW} {text}{Fore.RESET}"
    )


def debug(text: str):
    print(f"[{Fore.CYAN}{Style.BRIGHT}DEBUG{Style.RESET_ALL}]: {text}")


print("\n" * 30)
print("=" * 30 + " Starting " + "=" * 30)

mapping = Map()
target = Coordinate(100, 350, np.pi)
success = False
navigator = Navigation()
navigator.wait_for_activation()
mapping.keypoint(target)
mapping.show()

while True:

    if not success:
        navigator.move_to(target)
        success = True
        debug("Success")
    pass
