"""Пространственное подавление заградительной помехи + CA-CFAR (numpy-эталон)."""
from .cfar import CaCfarDetector, Detection
from .nuller import NullerReport, SubspaceNuller

__all__ = ["SubspaceNuller", "NullerReport", "CaCfarDetector", "Detection"]
