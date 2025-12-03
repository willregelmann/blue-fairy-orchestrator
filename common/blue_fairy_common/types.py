"""Type definitions for Blue Fairy configuration"""

from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from blue_fairy_common.plugin_types import PluginConfig


class AgentStatus(str, Enum):
    """Agent lifecycle states.

    State machine ensures agents are always in a known, valid state.
    """
    PENDING = "pending"
    PROVISIONING = "provisioning"
    STARTING = "starting"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UPGRADING = "upgrading"
    REMOVING = "removing"


VALID_TRANSITIONS: dict[AgentStatus, list[AgentStatus]] = {
    AgentStatus.PENDING: [AgentStatus.PROVISIONING, AgentStatus.REMOVING],
    AgentStatus.PROVISIONING: [AgentStatus.STARTING, AgentStatus.FAILED, AgentStatus.REMOVING],
    AgentStatus.STARTING: [AgentStatus.INITIALIZING, AgentStatus.FAILED, AgentStatus.REMOVING],
    AgentStatus.INITIALIZING: [AgentStatus.RUNNING, AgentStatus.FAILED, AgentStatus.REMOVING],
    AgentStatus.RUNNING: [AgentStatus.STOPPING, AgentStatus.FAILED, AgentStatus.REMOVING, AgentStatus.UPGRADING],
    AgentStatus.STOPPING: [AgentStatus.STOPPED, AgentStatus.FAILED],
    AgentStatus.STOPPED: [AgentStatus.STARTING, AgentStatus.REMOVING, AgentStatus.UPGRADING],
    AgentStatus.FAILED: [AgentStatus.STARTING, AgentStatus.REMOVING, AgentStatus.UPGRADING],
    AgentStatus.UPGRADING: [AgentStatus.RUNNING, AgentStatus.FAILED],
    AgentStatus.REMOVING: [],
}


class InvalidTransition(Exception):
    """Raised when attempting an invalid state transition"""
    pass


def validate_transition(current: AgentStatus, new: AgentStatus) -> None:
    """Validate a state transition.

    Raises:
        InvalidTransition: If transition is not valid
    """
    if new not in VALID_TRANSITIONS[current]:
        valid = [s.value for s in VALID_TRANSITIONS[current]]
        raise InvalidTransition(
            f"Cannot transition from {current.value} to {new.value}. "
            f"Valid transitions: {valid}"
        )


class ResourceConfig(BaseModel):
    """Container resource limits"""
    memory: Optional[str] = None  # e.g., "512Mi", "1Gi"
    cpu: Optional[str] = None     # e.g., "0.5", "1.0"


class PromptConfig(BaseModel):
    """Agent prompt configuration"""
    text: Optional[str] = None
    file: Optional[str] = None

    @field_validator('text', 'file')
    @classmethod
    def validate_prompt_xor(cls, v, info):
        """Ensure exactly one of text or file is set"""
        # This will be called for each field, but we need to check both
        # We'll do final validation in model_validator
        return v

    def model_post_init(self, __context):
        """Validate that exactly one of text or file is set"""
        if (self.text is None) == (self.file is None):
            raise ValueError("prompt must have either text or file, not both")


class MemoryConfig(BaseModel):
    """Agent memory configuration"""
    enabled: bool = False
    backend: Literal["mem0", "zep"] = "mem0"  # Memory backend to use
    path: str = "/data/memory"  # Container path for memory storage
    remem_enabled: bool = False  # Enable Think-Act-Refine memory evolution


class ActionQuotaConfig(BaseModel):
    """Action quota configuration for rate limiting agent messages"""
    limit: int = Field(gt=0, description="Max messages per window")
    window_seconds: int = Field(gt=0, description="Rolling window duration in seconds")
    low_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Warn agent when remaining <= this fraction"
    )


class ObservabilityEvents(BaseModel):
    """Event types to collect for Level 2 observability"""
    decisions: bool = True
    memory_retrievals: bool = True
    communications: bool = True
    reasoning: bool = False  # Very verbose, opt-in


class ObservabilityConfig(BaseModel):
    """Observability configuration for agents"""
    level: Literal[0, 1, 2] = Field(
        default=0,
        description="Observability level: 0=private, 1=metrics only, 2=full participation"
    )
    self_reflection_enabled: bool = Field(
        default=False,
        description="Allow agent to introspect its own telemetry via MCP"
    )
    events: ObservabilityEvents = Field(
        default_factory=ObservabilityEvents,
        description="Event types to collect (Level 2 only)"
    )


# ============================================================================
# Subroutines Configuration
# ============================================================================


class SleepConfig(BaseModel):
    """Configuration for harness-initiated sleep cycles"""
    enabled: bool = True
    idle_threshold_minutes: int = Field(
        default=30,
        ge=1,
        description="Sleep after N minutes of no messages"
    )
    activity_threshold_messages: int = Field(
        default=50,
        ge=1,
        description="Sleep after N messages processed"
    )
    schedule: Optional[str] = Field(
        default=None,
        description="Optional cron-like schedule for circadian patterns"
    )


class ConsolidateMemoryConfig(BaseModel):
    """Configuration for memory consolidation subroutine"""
    enabled: bool = True
    max_memories_per_run: int = Field(
        default=100,
        ge=1,
        description="Sample limit for large memory sets"
    )


class DreamConfig(BaseModel):
    """Configuration for dream subroutine"""
    enabled: bool = True
    max_duration_minutes: int = Field(
        default=5,
        ge=1,
        description="Timeout for dream subroutine"
    )
    memory_sample_size: int = Field(
        default=20,
        ge=1,
        description="How many memories to seed dream content"
    )


class SubroutinesConfig(BaseModel):
    """Configuration for agent subroutines (internal cognitive processes)"""
    enabled: bool = Field(
        default=False,
        description="Master switch for subroutines"
    )
    sleep: SleepConfig = Field(
        default_factory=SleepConfig,
        description="Harness-initiated sleep cycle settings"
    )
    consolidate_memory: ConsolidateMemoryConfig = Field(
        default_factory=ConsolidateMemoryConfig,
        description="Memory consolidation subroutine settings"
    )
    dream: DreamConfig = Field(
        default_factory=DreamConfig,
        description="Dream subroutine settings"
    )


class McpServerConfig(BaseModel):
    """Configuration for an external MCP server"""
    type: Literal["stdio", "sse", "http"]

    # For stdio transport
    command: Optional[str] = None
    args: Optional[list[str]] = Field(default_factory=list)

    # For HTTP/SSE transport
    url: Optional[str] = None

    # Environment variables (for stdio) or headers (for HTTP)
    env: Optional[dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_transport_fields(self):
        if self.type == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires 'command'")
        elif self.type in ("sse", "http"):
            if not self.url:
                raise ValueError(f"{self.type} transport requires 'url'")
        return self


class AgentTemplate(BaseModel):
    """Agent template configuration"""
    image: str
    model: str
    prompt: PromptConfig
    resources: Optional[ResourceConfig] = None
    memory: Optional[MemoryConfig] = None
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability and consent configuration"
    )
    action_quota: Optional[ActionQuotaConfig] = Field(
        default=None,
        description="Action quota for rate limiting messages (None = unlimited)"
    )
    subroutines: Optional[SubroutinesConfig] = Field(
        default=None,
        description="Subroutines configuration for internal cognitive processes (None = disabled)"
    )
    mcp_servers: Optional[dict[str, McpServerConfig]] = Field(
        default=None,
        description="External MCP servers for agent tool access"
    )
    plugins: list[PluginConfig] = Field(default_factory=list)


class AgentRuntimeConfig(BaseModel):
    """Config returned by supervisor API, consumed by harness.

    This is the contract between supervisor and harness.
    Both modules import this type for type safety.
    """
    agent_id: str
    model: str
    prompt_text: str
    memory_enabled: bool = False
    memory_backend: Literal["mem0", "zep"] = "mem0"
    remem_enabled: bool = False  # Enable Think-Act-Refine memory evolution
    action_quota_limit: Optional[int] = None
    action_quota_window: Optional[int] = None
    action_quota_low_threshold: float = 0.3
    subroutines_enabled: bool = False
    subroutines_config: Optional[dict] = None  # Serialized SubroutinesConfig
    mcp_servers: Optional[dict[str, dict]] = None  # Serialized McpServerConfig
    plugins: list[PluginConfig] = Field(default_factory=list)  # Plugin configurations


class Config(BaseModel):
    """Main Blue Fairy configuration"""
    version: str
    agents: dict[str, AgentTemplate] = Field(default_factory=dict)

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Ensure version is supported"""
        if v not in ["0.1", "0.2"]:
            raise ValueError(f"Unsupported version: {v}")
        return v

    @field_validator('agents')
    @classmethod
    def validate_agents(cls, v):
        """Ensure at least one agent is defined"""
        if not v:
            raise ValueError("At least one agent template is required")
        return v
