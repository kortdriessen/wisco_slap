# Blessed top-level API
from .core.SlapData import SlapData

# Make subpackages appear under `wisco_slap.`
from . import img as img

__all__ = ['SlapData', 'img']