"""core/motion -- кинематика движения цели (P1: cv/Markov/вираж/разгон/манёвр + проекция в бины)."""
from .kinematics import Kinematics, KinematicsSample
from .models import (
    ConstantAccel,
    ConstantVelocity,
    CoordinatedTurn,
    MarkovDrift,
    MotionModel,
    WeavingManeuver,
)
from .state import TargetState

__all__ = [
    "TargetState",
    "MotionModel", "ConstantVelocity", "MarkovDrift", "CoordinatedTurn", "ConstantAccel",
    "WeavingManeuver",
    "Kinematics", "KinematicsSample",
]
