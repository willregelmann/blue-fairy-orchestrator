"""Configuration loading and validation"""

from pathlib import Path
from typing import Union
import yaml
from pydantic import ValidationError

from .types import Config


class ConfigError(Exception):
    """Configuration loading or validation error"""
    pass


class ConfigLoader:
    """Load and validate Blue Fairy configuration files"""

    def load(self, config_path: Union[str, Path]) -> Config:
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}")

        return self.load_from_dict(data, base_path=config_path.parent)

    def load_from_dict(self, data: dict, base_path: Path = None) -> Config:
        """Load configuration from dictionary"""
        # Validate required fields
        if "version" not in data:
            raise ConfigError("version is required")

        if "agents" not in data:
            raise ConfigError("agents section is required")

        # Resolve file-based prompts
        if base_path:
            for agent_name, agent_data in data.get("agents", {}).items():
                if "prompt" in agent_data and "file" in agent_data["prompt"]:
                    prompt_file = agent_data["prompt"]["file"]
                    if not Path(prompt_file).is_absolute():
                        agent_data["prompt"]["file"] = str(base_path / prompt_file)

        # Validate with pydantic
        try:
            return Config(**data)
        except ValidationError as e:
            # Convert pydantic errors to ConfigError
            errors = []
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{field}: {msg}")
            raise ConfigError("Validation errors:\n" + "\n".join(errors))
