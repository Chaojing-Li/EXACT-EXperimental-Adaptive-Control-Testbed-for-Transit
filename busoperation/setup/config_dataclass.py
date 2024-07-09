from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TerminalNodeGeometry:
    x: float
    y: float


@dataclass
class TerminalNodeOperation:
    ...


@dataclass(frozen=True)
class StopNodeGeometry:
    x: float
    y: float
    berth_num: int


@dataclass(frozen=True)
class StopNodeOperation:
    is_alight: bool
    queue_rule: Literal['FO', 'FIFO']
    board_truncation: Literal['arrival', 'rtd']


@dataclass(frozen=True)
class PaxOperation:
    pax_arrival_type: Literal['deterministic', 'poisson']
    pax_board_time_mean: float
    pax_board_time_std: float
    pax_board_time_type: Literal['deterministic', 'normal']
    pax_alight_time_mean: Optional[float]
    pax_alight_time_std: Optional[float]
    pax_alight_time_type: Optional[Literal['deterministic', 'normal']]


@dataclass(frozen=True)
class LinkGeometry:
    head_node: str
    tail_node: str
    x_head: float
    y_head: float
    length: float


@dataclass
class LinkDistribution:
    tt_mean: float
    tt_cv: float
    tt_type: str
