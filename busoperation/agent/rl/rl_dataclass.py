from typing import List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    time: int
    route_id: str
    bus_id: str
    stop_id: str
    state: List[float]
    action: float
    reward: Optional[float]
