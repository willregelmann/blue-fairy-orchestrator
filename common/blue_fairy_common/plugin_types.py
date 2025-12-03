"""Plugin configuration types for Blue Fairy agents.

Plugins follow Claude Code conventions with extensions:
- "listener" hook type for long-running event sources
- "command" hook type for synchronous bootstrap commands
- "mcp" hook type for MCP server processes
"""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class PluginConfig(BaseModel):
    """Configuration for a plugin in agent template.

    Attributes:
        name: Plugin package name (e.g., "blue-fairy-matrix")
        config: Plugin-specific configuration options
    """
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class CommandHookConfig(BaseModel):
    """Configuration for a command hook (synchronous, blocking).

    Command hooks run during SessionStart before MCP servers and listeners.
    Used for installing dependencies or other setup tasks.

    Attributes:
        type: Must be "command"
        command: Executable to run
        args: Command arguments
        timeout: Max seconds to wait (default 120)
        cache_key: If provided, skip if already run (persists in /data/.plugin-cache/)
        env: Additional environment variables
    """
    type: Literal["command"] = "command"
    command: str
    args: list[str] = Field(default_factory=list)
    timeout: int = 120
    cache_key: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)


class McpHookConfig(BaseModel):
    """Configuration for an MCP server hook.

    MCP hooks start background processes that provide tools via MCP protocol.

    Attributes:
        type: Must be "mcp"
        command: Executable to run
        args: Command arguments
        env: Additional environment variables
    """
    type: Literal["mcp"] = "mcp"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ListenerConfig(BaseModel):
    """Configuration for a listener hook.

    Listeners are long-running processes that watch for external events
    and emit them as newline-delimited JSON to stdout.

    Attributes:
        type: Must be "listener"
        command: Executable to run
        args: Command arguments
        config: Configuration passed as environment variables
    """
    type: Literal["listener"] = "listener"
    command: str
    args: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


# Union type for all hook configurations
HookConfig = CommandHookConfig | McpHookConfig | ListenerConfig


class PluginManifest(BaseModel):
    """Plugin manifest from plugin.json.

    Follows Claude Code plugin schema.

    Attributes:
        name: Plugin identifier
        version: Semantic version
        description: Human-readable description
        mcp_servers: MCP server definitions (command, args)
    """
    name: str
    version: str
    description: str = ""
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
