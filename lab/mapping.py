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

REGISTRY_SCALE_X = 19
REGISTRY_SCALE_Y = 20
REGISTRY_ROTATION = np.deg2rad(90)
REGISTRY_TX = 206
REGISTRY_TY = 145


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
        print(f"{x}, {y} -> {result[1], result[0]}")
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


def warn(text: str) -> None:
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}{Fore.YELLOW} {text}{Fore.RESET}"
    )


def debug(text: str):
    print(f"[{Fore.CYAN}{Style.BRIGHT}DEBUG{Style.RESET_ALL}]: {text}")


print("\n" * 30)
print("=" * 30 + " Starting " + "=" * 30)

mapping = Map()

while True:
    HAL.setV(0.2)
    HAL.setW(0.4)
    x, y = HAL.getPose3d().x, HAL.getPose3d().y
    coord = Coordinate.from_world_coordinates(x, y)
    mapping.keypoint(coord)
    mapping.show()
