import enum
from dataclasses import dataclass


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


@dataclass
class SettingsDataclass:
    host: str = "127.0.0.1"
    port: int = 8888
    workers_count: int = 1
    reload: bool = False
    environment: str = "dev"
    log_level: LogLevel = LogLevel.INFO


settings = SettingsDataclass()
