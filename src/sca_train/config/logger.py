from enum import Enum


class SCATrainingVerbosityLevel(int, Enum):
    QUIET = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4