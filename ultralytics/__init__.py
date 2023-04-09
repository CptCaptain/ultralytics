# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = '8.0.71'

from ultralytics.hub import start
from ultralytics.yolo.engine.model import YOLO, CNS_YOLO
from ultralytics.yolo.utils.checks import check_yolo as checks

__all__ = '__version__', 'YOLO', 'CNS_YOLO', 'checks', 'start'  # allow simpler import

