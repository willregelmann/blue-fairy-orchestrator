"""Blue Fairy Common - Shared configuration and types"""

__version__ = "0.1.0"

from .config import ConfigLoader, ConfigError
from .types import Config, AgentTemplate, PromptConfig, ResourceConfig

__all__ = [
    "ConfigLoader",
    "ConfigError",
    "Config",
    "AgentTemplate",
    "PromptConfig",
    "ResourceConfig",
]
