from ompl import base as ob  # type: ignore
from ompl import geometric as og  # type: ignore

import numpy as np
from colorama import Fore, Style
from functools import partial
from typing import TypedDict, Any, Optional
import HAL  # type: ignore # noqa
import GUI  # type: ignore # noqa


class Coordinate(TypedDict):
    col: int
    row: int
    yaw: float


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
        start().setX(src["row"])
        start().setY(src["col"])
        start().setYaw(src["yaw"])

        goal = ob.State(space)
        goal().setX(dst["row"])
        goal().setY(dst["col"])
        goal().setYaw(dst["yaw"])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = og.RRTConnect(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solution = planner.solve(1.0)

        if solution:
            printed_mat = pdef.getSolutionPath().printAsMatrix()
            return self._parse_printed_matrix(printed_mat)

        return None

    @staticmethod
    def _is_state_valid(
        _mapping: np.ndarray, _invalid_value: float, state: Any
    ) -> bool:
        x = int(state.getX())
        y = int(state.getY())
        try:
            result = _mapping[x, y] != _invalid_value
            debug(f"Checking values {x}, {y} -> {result}")
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

            row, col, yaw, *_ = words
            coord: Coordinate = {"row": float(row), "col": float(col), "yaw": float(yaw)}
            result.append(coord)

        return result


def warn(text: str) -> None:
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}{Fore.YELLOW} {text}{Fore.RESET}"
    )


def debug(text: str):
    print(f"[{Fore.CYAN}{Style.BRIGHT}DEBUG{Style.RESET_ALL}]: {text}")


print("\n" * 30)
print("=" * 30 + " Starting " + "=" * 30)

mapping = np.zeros((10, 10))
mapping[5:10, 5:8] = -1

planner = OmplManager(mapping, invalid_value=-1)
src: Coordinate = {"row": 0, "col": 0, "yaw": 0}
dst: Coordinate = {"row": 9, "col": 9, "yaw": 0}
result = planner.plan(src, dst)
# print(f"{result=}")

for idx, coord in enumerate(result, start=1):
    mapping[int(coord["row"]), int(coord["col"])] = idx

print(mapping)

while True:
    pass
