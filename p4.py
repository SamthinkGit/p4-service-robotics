"""
Warehouse Robot Navigation System
=================================
This module implements a navigation system for a warehouse robot.
"""

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

# ==================================================================
# [IMPORTANT NOTE]: Due to the size and number of comments and code,
# important functions are indicated with this identifier:
#
# # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
#
# Make your life easier by going direclty to those functions :D
#
# ==================================================================


# =================== 🛠️  CONSTANTS 🛠️  =======================


class ColorCode(Enum):
    """Defines color codes used for map visualization and robot state
    representation."""

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
    """Represents the different operational states of the FSM"""

    PLANNING = auto()
    GOING_TO_TARGET = auto()
    LIFT = auto()
    PLANNING_TO_UNLIFT = auto()
    CARRYING = auto()
    FINISH = auto()


OCCUPIED_PIXEL = 0.9
FREE_PIXEL = 0.1
KEYPOINT_SIZE = 2
KEYPOINT_COLOR = ColorCode.YELLOW.value
KEYPOINT_BORDER_COLOR = ColorCode.ORANGE.value

REGISTRY_SCALE_X = 21.5
REGISTRY_SCALE_Y = 20


PRINT_ONLY_VALID_PATH = True
REGISTRY_ROTATION = np.deg2rad(90)
REGISTRY_TX = 207
REGISTRY_TY = 145
ROTATION_FORCE = 0.2
ROTATION_FORCE_FORWARDING = 0.8
VELOCITY = 0.2

YAW_PRECISSION = 0.1
YAW_PRECISSION_FORWARDING = 0.05
FINAL_YAW_PRECISSION = 0.05

DISTANCE_PRECISSION = 5
VECTOR_PRECISSION = DISTANCE_PRECISSION / 2
ROBOT_YAW_DISPLACEMENT = 90

TIME_TO_SOLVE = 90
LINE_THICKNESS = 1
ARROW_LENGTH = 30
LIFTING_ANGLE = 90

ompl_last_state: Optional[Any] = None


# =================== ✨ Core Modules ✨ =======================
@dataclass
class Vector:
    """Represents a directional vector with magnitude and angle. It is mainly
    focused to be used with local navigation"""

    module: float
    alpha: float

    @classmethod
    def from_coordinates(cls, src: "Coordinate", dst: "Coordinate") -> "Vector":
        """Creates a Vector from two Coordinate points by calculating the
        distance and angle between them."""

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
    """Wrapper for unifiying OpenCV, axis-based and robot coordinates as a data
    structure"""

    row: int
    col: int
    yaw: float = field(default=0.0)

    @classmethod
    def from_world_coordinates(cls, x: float, y: float) -> "Coordinate":
        # ===================================
        # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
        # ===================================
        """Converts the coordinates obtained from the robot into col-row based representation"""
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
    """Manager for simplifying path planning using the OMPL library."""

    def __init__(self, mapping: np.ndarray, invalid_value: int):
        self.mapping = mapping
        self.invalid_value = invalid_value

    def plan(self, src: Coordinate, dst: Coordinate) -> Optional[list[Coordinate]]:
        # ===================================
        # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
        # ===================================
        """Plans a path from a source Coordinate to a destination Coordinate
        using the RRTConnect algorithm from the OMPL library."""
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
        # ===================================
        # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
        # ===================================
        """Checks if a given state is valid by ensuring it does not collide with
        obstacles in the mapping."""

        global MAP, ompl_last_state

        # Get the coordinates
        x = int(state.getX())
        y = int(state.getY())

        # If it is the first vector, just save it
        if ompl_last_state is None:
            save_yaw = float(state.getYaw())

        # If there are more, compute the yaw from the previous to the current
        else:
            new_state = Coordinate(row=x, col=y, yaw=None)
            vec = Vector.from_coordinates(ompl_last_state, new_state)
            save_yaw = vec.alpha

        # If the yaw could't be resolved fallback to 0
        if save_yaw is None:
            save_yaw = 0

        # Print the obtained vector
        ompl_last_state = Coordinate(row=x, col=y, yaw=save_yaw)
        MAP.arrow(ompl_last_state, color=ColorCode.YELLOW.value)

        # We rotate the yaw 90deg because we consider 0 at 90º from the X-axis
        rotation = np.rad2deg(save_yaw) - 90
        try:

            # Build a mask and search if there are any obstacles
            mask = np.zeros(_mapping.shape, dtype=np.uint8)
            draw_rectangle(mask, 1, (x, y), ROBOT_SIZE_Y, ROBOT_SIZE_X, rotation, True)
            masked_values = _mapping[mask == 1]
            result = not np.any(masked_values == _invalid_value)

            # Print the result and return it
            color = ColorCode.GREEN.value if result else ColorCode.RED.value
            draw_rectangle(
                MAP.gui, color, (x, y), ROBOT_SIZE_Y, ROBOT_SIZE_X, rotation, False
            )
            MAP.show()

            return result

        except IndexError:
            warn(f"Trying to access to value {x}, {y} which is not present on the map")
            return False

    @staticmethod
    def _parse_printed_matrix(printed_mat: str) -> list[Coordinate]:
        """Parses a printed matrix string from the OMPL library into a list of
        Coordinate instances."""
        result = []
        for line in printed_mat.splitlines():

            # We assume the matrix will always contain (x, y, yaw)
            if len(words := line.split(" ")) < 3:
                continue

            row, col, yaw = [int(float(val)) for val in words[:3]]
            coord = Coordinate(row, col, yaw)
            result.append(coord)

        return result


class Map:
    """Represents the warehouse map and provides utilities for visualization and editing."""

    def __init__(self):
        # ===================================
        # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
        # ===================================
        self.map = GUI.getMap("/resources/exercises/amazon_warehouse/images/map.png")

        gui = (
            np.ones((self.map.shape[0], self.map.shape[1]), dtype=np.uint8)
            * ColorCode.GRAY.value
        )
        gui[np.all(self.map > [OCCUPIED_PIXEL] * 3, axis=-1)] = ColorCode.WHITE.value
        gui[np.all(self.map < [FREE_PIXEL] * 3, axis=-1)] = ColorCode.BLACK.value
        self.parsed_map = deepcopy(gui)
        self.gui = gui

    def erase(self, center: tuple[int, int], size: tuple[int, int], rotation):
        """Erases a rectangular area on the map's GUI representation."""
        erase_func = partial(
            draw_rectangle,
            color=ColorCode.WHITE.value,
            center=center,
            width=size[0],
            height=size[1],
            rotation=rotation,
            filled=True,
        )
        erase_func(self.parsed_map)
        erase_func(self.gui)

    def show(self):
        """Displays the current state of the map GUI."""
        GUI.showNumpy(self.gui)

    def clean(self):
        """Resets the map to its initial state."""
        self.__init__()

    # ================= ✒️ GUI Drawing Methods =========================

    def keypoint(self, coord: Coordinate, color=KEYPOINT_COLOR) -> None:
        """Draws a squared keypoint on the GUI"""
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

    def circle(
        self, coord: Coordinate, radius: float, color: int = ColorCode.BLACK.value
    ) -> None:
        """Draws a circle on the map at the specified coordinate."""

        center = (coord.col, coord.row)
        cv2.circle(self.gui, center, radius, color, LINE_THICKNESS)

    def connect(
        self, src: Coordinate, dst: Coordinate, color: int = ColorCode.RED.value
    ) -> None:
        """Draws a line connecting two coordinates on the map."""
        cv2.line(
            self.gui, (src.col, src.row), (dst.col, dst.row), color, LINE_THICKNESS
        )

    def arrow(self, coord: Coordinate, color: int = ColorCode.RED.value) -> None:
        """Draws an arrow on the map at the specified coordinate indicating the
        direction of movement."""
        if coord.yaw is None:
            raise ValueError("Yaw error cannot be none for drawing an arrow")

        end_point = (
            int(coord.col + ARROW_LENGTH * np.cos(coord.yaw)),
            int(coord.row - ARROW_LENGTH * np.sin(coord.yaw)),
        )

        cv2.arrowedLine(
            self.gui, (coord.col, coord.row), end_point, color, LINE_THICKNESS
        )

    # ==============================================================


class Navigation:
    """Handles robot navigation and movement."""

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
        """Waits until the navigation system is activated and ready for use."""
        while self.yaw is None:
            pass
        debug("Navigator is now active")

    def move_to(self, dst: Coordinate, ignore_yaw: bool = True) -> None:
        # ===================================
        # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
        # ===================================
        """Moves the robot to the specified destination coordinate, adjusting
        yaw and distance as needed."""

        # Rotate the robot to the next point
        yaw_error = float("inf")
        while abs(yaw_error) > YAW_PRECISSION:
            src = self.coords
            target = Vector.from_coordinates(src, dst)
            yaw_error = shortest_angle_distance_radians(src.yaw, target.alpha)
            HAL.setW(yaw_error * ROTATION_FORCE)

        # Go to the next point
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

        if ignore_yaw:
            yaw_error = 0

        # Reorient as specified (deactivated by default)
        while abs(yaw_error) > FINAL_YAW_PRECISSION:
            src = self.coords
            yaw_error = shortest_angle_distance_radians(src.yaw, dst.yaw)
            HAL.setW(yaw_error * ROTATION_FORCE)

        HAL.setW(0)


def draw_rectangle(
    dst,
    color: int,
    center: tuple[int, int],
    width,
    height,
    rotation=0,
    filled: bool = False,
) -> None:
    """Draws a rectangle on the specified destination with given properties."""
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


# ======================= DEBUGGING =============================
def warn(text: str) -> None:
    """Outputs a warning message to the console with formatting."""
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}{Fore.YELLOW} {text}{Fore.RESET}"
    )


def debug(text: str):
    """Outputs a debug message to the console with formatting."""
    print(f"[{Fore.CYAN}{Style.BRIGHT}DEBUG{Style.RESET_ALL}]: {text}")


# ======================= Initialization =============================

print("\n" * 30)
print("=" * 30 + " Starting " + "=" * 30)

shelves = [
    (3.728, 0.579),  # Shelve 1
    (3.728, -1.242),  # Shelve 2
    (3.728, -3.039),  # Shelve 3
    (3.728, -4.827),  # Shelve 4
    (3.728, -6.781),  # Shelve 5
    (3.728, -8.665),  # Shelve 6
]

# ====================================================
# Target Selection
shelve_id = 5
move_to = Coordinate(row=215, col=350, yaw=0.0)
# ===================================================


x, y = shelves[shelve_id - 1]
target = Coordinate.from_world_coordinates(x, y)
target.yaw = np.deg2rad(LIFTING_ANGLE)
second_target = move_to

mapping = Map()
MAP = mapping

navigator = Navigation()
navigator.wait_for_activation()
mapping.keypoint(target)
mapping.show()
ompl_man = OmplManager(mapping=mapping.parsed_map, invalid_value=ColorCode.BLACK.value)

# ======= ROBOT SIZE (NON-CONSTANT) ========
ROBOT_SIZE_Y = 12
ROBOT_SIZE_X = 12

state = States.PLANNING

while True:

    match state:

        case States.PLANNING:

            mapping.keypoint(target, color=ColorCode.GRAY.value)
            plan = ompl_man.plan(navigator.coords, target)

            for idx, coord in enumerate(plan[:-1]):
                debug(f"{idx}. {coord}")
                mapping.connect(coord, plan[idx + 1])
                mapping.circle(coord, radius=10, color=ColorCode.YELLOW.value)

            mapping.arrow(target, color=ColorCode.VIOLET.value)
            mapping.show()

            state = States.GOING_TO_TARGET

        case States.GOING_TO_TARGET:

            for coord in plan[1:-1]:
                mapping.keypoint(coord, color=ColorCode.BLACK.value)
                mapping.show()
                navigator.move_to(coord)

            navigator.move_to(target, ignore_yaw=False)

            state = States.LIFT

        case States.LIFT:
            ROBOT_SIZE_Y = 47
            ROBOT_SIZE_X = 24

            HAL.lift()
            mapping.clean()
            coords = navigator.coords
            mapping.erase(
                (coords.row, coords.col),
                (ROBOT_SIZE_Y, ROBOT_SIZE_X),
                rotation=0,
            )
            mapping.show()
            ompl_man = OmplManager(
                mapping=mapping.parsed_map, invalid_value=ColorCode.BLACK.value
            )

            debug("Lifting Completed")
            state = States.PLANNING_TO_UNLIFT

        case States.PLANNING_TO_UNLIFT:

            mapping.keypoint(second_target, color=ColorCode.GRAY.value)
            plan = ompl_man.plan(navigator.coords, second_target)

            for idx, coord in enumerate(plan[:-1]):
                debug(f"{idx}. {coord}")
                mapping.connect(coord, plan[idx + 1])
                mapping.circle(coord, radius=27, color=ColorCode.YELLOW.value)

            mapping.arrow(second_target, color=ColorCode.VIOLET.value)
            mapping.show()

            state = States.CARRYING

        case States.CARRYING:

            for coord in plan[1:-1]:
                mapping.keypoint(coord, color=ColorCode.BLACK.value)
                mapping.show()
                navigator.move_to(coord)

            navigator.move_to(second_target, ignore_yaw=False)

            HAL.putdown()
            mapping.clean()
            mapping.show()
            debug("Routine Completed")
            state = States.FINISH
