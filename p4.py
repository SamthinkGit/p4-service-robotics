import HAL  # type: ignore # noqa
import GUI  # type: ignore # noqa
import cv2
from enum import Enum, auto
from dataclasses import dataclass, field
from colorama import Fore, Style
from copy import deepcopy
import numpy as np

from ompl import base as ob  # type: ignore
from ompl import geometric as og  # type: ignore
from functools import partial
from typing import Any, Optional


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


class States(Enum):
    PLANNING = auto()
    GOING_TO_TARGET = auto()
    FINISH = auto()


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
ROTATION_FORCE_FORWARDING = 0.5
VELOCITY = 0.2

YAW_PRECISSION = 0.1
YAW_PRECISSION_FORWARDING = 0.05
DISTANCE_PRECISSION = 5
VECTOR_PRECISSION = DISTANCE_PRECISSION / 2
ROBOT_YAW_DISPLACEMENT = 90

TIME_TO_SOLVE = 90
CIRCLE_SIZE = 3
LINE_THICKNESS = 1
ARROW_LENGTH = 30


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

        alpha = np.arctan2(dy, dx)

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


class OmplManager:

    def __init__(self, mapping: np.ndarray, invalid_value: int):
        self.mapping = mapping
        self.invalid_value = invalid_value

    def plan(self, src: Coordinate, dst: Coordinate) -> Optional[list[Coordinate]]:
        space = ob.SE2StateSpace()

        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0)
        bounds.setLow(1, 0)
        bounds.setHigh(0, self.mapping.shape[0])
        bounds.setHigh(1, self.mapping.shape[1])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        validation_func = partial(
            self._is_state_valid, self.mapping, self.invalid_value
        )
        si.setStateValidityChecker(ob.StateValidityCheckerFn(validation_func))

        start = ob.State(space)
        start().setX(float(src.row))
        start().setY(float(src.col))
        start().setYaw(src.yaw)

        goal = ob.State(space)
        goal().setX(float(dst.row))
        goal().setY(float(dst.col))
        goal().setYaw(dst.yaw)

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = og.RRTConnect(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        debug("Planning...")
        solution = planner.solve(TIME_TO_SOLVE)

        if solution:
            debug("Solution Found")
            printed_mat = pdef.getSolutionPath().printAsMatrix()
            return self._parse_printed_matrix(printed_mat)

        debug(f"Planning Failed: {solution}")
        return None

    @staticmethod
    def _is_state_valid(
        _mapping: np.ndarray, _invalid_value: float, state: Any
    ) -> bool:
        global MAP

        x = int(state.getX())
        y = int(state.getY())
        try:
            result = _mapping[x, y] != _invalid_value
            color = ColorCode.GREEN.value if result else ColorCode.RED.value
            MAP.circle(Coordinate(x, y), color)
            MAP.show()

            return result

        except IndexError:
            warn(f"Trying to access to value {x}, {y} which is not present on the map")
            return False

    @staticmethod
    def _parse_printed_matrix(printed_mat: str) -> list[Coordinate]:
        result = []
        for line in printed_mat.splitlines():

            if len(words := line.split(" ")) < 3:
                continue

            row, col, yaw = [int(float(val)) for val in words[:3]]
            coord = Coordinate(row, col, yaw)
            result.append(coord)

        return result


class Map:

    def __init__(self):
        self.map = GUI.getMap("/resources/exercises/amazon_warehouse/images/map.png")

        gui = (
            np.ones((self.map.shape[0], self.map.shape[1]), dtype=np.uint8)
            * ColorCode.GRAY.value
        )
        gui[np.all(self.map > [OCCUPIED_PIXEL] * 3, axis=-1)] = ColorCode.WHITE.value
        gui[np.all(self.map < [FREE_PIXEL] * 3, axis=-1)] = ColorCode.BLACK.value
        self.parsed_map = deepcopy(gui)
        self.gui = gui

    def show(self):
        GUI.showNumpy(self.gui)

    def keypoint(self, coord: Coordinate, color=KEYPOINT_COLOR):
        if not isinstance(color, int):
            raise ValueError("Invalid color received")
        border_size = KEYPOINT_SIZE + 1
        self.gui[
            coord.row - border_size : coord.row + border_size,  # noqa
            coord.col - border_size : coord.col + border_size,  # noqa
        ] = KEYPOINT_BORDER_COLOR

        self.gui[
            coord.row - KEYPOINT_SIZE : coord.row + KEYPOINT_SIZE,  # noqa
            coord.col - KEYPOINT_SIZE : coord.col + KEYPOINT_SIZE,  # noqa
        ] = color

    def circle(self, coord: Coordinate, color: int = ColorCode.BLACK.value):
        center = (coord.col, coord.row)
        radius = CIRCLE_SIZE
        cv2.circle(self.gui, center, radius, color, LINE_THICKNESS)

    def connect(
        self, src: Coordinate, dst: Coordinate, color: int = ColorCode.RED.value
    ):
        cv2.line(
            self.gui, (src.col, src.row), (dst.col, dst.row), color, LINE_THICKNESS
        )

    def arrow(self, coord: Coordinate, color: int = ColorCode.RED.value):
        if coord.yaw is None:
            raise ValueError("Yaw error cannot be none for drawing an arrow")

        end_point = (
            int(coord.col + ARROW_LENGTH * np.cos(coord.yaw)),
            int(coord.row - ARROW_LENGTH * np.sin(coord.yaw)),
        )

        cv2.arrowedLine(self.gui, (coord.col, coord.row), end_point, color, LINE_THICKNESS)

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
            # print(f"[{yaw_error:.2f}] {np.rad2deg(src.yaw):.2f} -> {np.rad2deg(target.alpha):.2f}")

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
        This function ensures that the distance is minimized br
    cy accounting for the
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
MAP = mapping

navigator = Navigation()
navigator.wait_for_activation()
target = Coordinate(row=100, col=300, yaw=0.0)
mapping.keypoint(target)
mapping.show()
ompl_man = OmplManager(mapping=mapping.parsed_map, invalid_value=ColorCode.BLACK.value)


state = States.PLANNING

while True:

    match state:

        case States.PLANNING:

            plan = ompl_man.plan(navigator.coords, target)

            for idx, coord in enumerate(plan[:-1]):
                debug(f"{idx}. {coord}")
                mapping.connect(coord, plan[idx + 1])
                mapping.arrow(plan[idx + 1])
            mapping.show()

            state = States.GOING_TO_TARGET

        case States.GOING_TO_TARGET:
            for coord in plan[1:]:
                mapping.keypoint(coord, color=ColorCode.BLACK.value)
                mapping.arrow(coord, color=ColorCode.BLACK.value)
                mapping.show()
                navigator.move_to(coord)
            state = States.FINISH
    pass
