"""Agent lifecycle management"""

import json
import os
import time
import tempfile
import requests
import shutil
from pathlib import Path
from typing import Optional

from blue_fairy_common import Config, AgentTemplate
from blue_fairy_common.types import AgentStatus, InvalidTransition
from .state import StateManager
from .docker_mgr import DockerManager
from .continuity import generate_continuity_secret, plant_continuity_secret, verify_continuity
from .deploy_keys import generate_ssh_keypair, GitHubDeployKeyManager

# Docker network service name for Matrix homeserver
# Used for container-to-container communication (agents in containers
# cannot reach localhost, but can reach other containers by service name)
MATRIX_CONTAINER_SERVICE_NAME = "blue-fairy-synapse:8008"


def load_api_key() -> str:
    """Load ANTHROPIC_API_KEY from environment or .env file

    Priority:
    1. ANTHROPIC_API_KEY environment variable
    2. .env file in project root
    3. .env file in ~/.blue-fairy/

    Returns:
        API key string

    Raises:
        ValueError if no API key found
    """
    # Check environment variable first
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return api_key

    # Try loading from .env files
    env_paths = [
        Path.cwd() / '.env',
        Path.home() / '.blue-fairy' / '.env',
    ]

    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith('ANTHROPIC_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    if api_key:
                        return api_key

    raise ValueError("ANTHROPIC_API_KEY not found in environment or .env files")


class LifecycleManager:
    """Manage agent spawning, initialization, and shutdown"""

    def __init__(
        self,
        state: StateManager,
        docker: DockerManager,
        config: Config,
        matrix_manager=None,
        deploy_key_manager=None
    ):
        self.state = state
        self.docker = docker
        self.config = config
        self.matrix_manager = matrix_manager
        self.deploy_key_manager = deploy_key_manager

    async def spawn_agent(
        self,
        template_name: str,
        agent_id: Optional[str] = None
    ) -> dict:
        """Spawn an agent from template using state machine transitions.

        State flow: PENDING → PROVISIONING → STARTING → INITIALIZING → RUNNING
        On failure: automatic rollback with transition to FAILED
        """
        print(f"[SPAWN] Starting agent spawn: template={template_name}", flush=True)

        # Get template
        if template_name not in self.config.agents:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.config.agents[template_name]

        # Generate agent_id if not provided
        if not agent_id:
            agent_id = self._generate_agent_id(template_name)

        try:
            # Phase 1: Create agent in PENDING state with complete config
            print(f"[SPAWN] Creating agent record in PENDING state", flush=True)
            self._create_pending_agent(agent_id, template_name, template)

            # Phase 2: Provision resources (memory dir, Matrix account)
            print(f"[SPAWN] Provisioning resources", flush=True)
            self.state.transition(agent_id, AgentStatus.PROVISIONING)
            resources = self._provision_resources(agent_id, template)

            # Provision Matrix account (async operation)
            if resources.get("matrix_account"):
                await self._provision_matrix_account(agent_id)

            # Store provisioned resources for rollback tracking
            self.state.set_provisioned_resources(agent_id, resources)

            # Phase 3: Start container
            print(f"[SPAWN] Starting container", flush=True)
            self.state.transition(agent_id, AgentStatus.STARTING)
            environment = self._build_environment(agent_id, template)
            agent_url = self._start_container(agent_id, template, environment)

            # Update config with host URL
            agent = self.state.get_agent(agent_id)
            agent_config = agent.config.copy()
            agent_config["host_url"] = agent_url
            self.state.update_agent_config(agent_id, agent_config)

            # Wait for container health check
            print(f"[SPAWN] Waiting for health check...", flush=True)
            if not await self._wait_for_health(agent.container_id, agent_url):
                raise Exception("Container failed to become healthy")

            # Phase 4: Initialize agent (choose name)
            print(f"[SPAWN] Initializing agent", flush=True)
            self.state.transition(agent_id, AgentStatus.INITIALIZING)
            name = self._initialize_agent(agent_id, agent_url)

            # Phase 5: Transition to RUNNING
            self.state.transition(agent_id, AgentStatus.RUNNING)
            print(f"[SPAWN] Agent '{name}' ({agent_id}) spawned successfully", flush=True)

            return {
                "agent_id": agent_id,
                "name": name,
                "status": "running"
            }

        except Exception as e:
            # Rollback on any failure
            error_message = str(e)
            print(f"[SPAWN] Spawn failed: {error_message}", flush=True)
            await self._rollback(agent_id, error_message)
            raise

    def stop_agent(self, identifier: str) -> dict:
        """Stop an agent by agent_id or name"""
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Validate current state allows stopping
        current_status = AgentStatus(agent.status)
        if current_status not in [AgentStatus.RUNNING, AgentStatus.INITIALIZING]:
            raise InvalidTransition(
                f"Cannot stop agent in {current_status.value} state. "
                f"Agent must be running or initializing."
            )

        # Transition to STOPPING
        self.state.transition(agent.agent_id, AgentStatus.STOPPING)

        try:
            # Stop container (preserves container for restart)
            self.docker.stop_container(agent.container_id)

            # Transition to STOPPED
            self.state.transition(agent.agent_id, AgentStatus.STOPPED)

            return {"status": "stopped", "agent_id": agent.agent_id}
        except Exception as e:
            # On failure, transition to FAILED
            self.state.transition(agent.agent_id, AgentStatus.FAILED)
            self.state.set_error(agent.agent_id, f"Stop failed: {e}")
            raise

    async def start_agent(self, identifier: str) -> dict:
        """Start a stopped agent by agent_id or name"""
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Validate current state allows starting
        current_status = AgentStatus(agent.status)
        if current_status not in [AgentStatus.STOPPED, AgentStatus.FAILED]:
            if current_status == AgentStatus.RUNNING:
                return {"status": "running", "message": "Agent already running", "agent_id": agent.agent_id}
            raise InvalidTransition(
                f"Cannot start agent in {current_status.value} state. "
                f"Agent must be stopped or failed."
            )

        # Transition to STARTING
        self.state.transition(agent.agent_id, AgentStatus.STARTING)

        try:
            # Start container
            self.docker.start_container(agent.container_id)

            # Get new port binding after restart
            container = self.docker.client.containers.get(agent.container_id)
            port_bindings = container.attrs['NetworkSettings']['Ports'].get('8080/tcp')
            if port_bindings:
                host_port = port_bindings[0]['HostPort']
                new_host_url = f"localhost:{host_port}"
            else:
                # Fallback to old IP-based approach if no port binding
                new_host_url = container.attrs['NetworkSettings']['Networks']['blue-fairy-net']['IPAddress'] + ":8080"

            # Update host URL in agent config
            agent_config = agent.config.copy()
            agent_config["host_url"] = new_host_url
            self.state.update_agent_config(agent.agent_id, agent_config)

            # Wait for health check
            print(f"[START] Waiting for health check...", flush=True)
            if not await self._wait_for_health(agent.container_id, new_host_url):
                raise Exception("Container failed health check after restart")

            # Transition through INITIALIZING (no re-initialization needed, agent already has name)
            self.state.transition(agent.agent_id, AgentStatus.INITIALIZING)

            # Transition to RUNNING
            self.state.transition(agent.agent_id, AgentStatus.RUNNING)

            return {
                "status": "running",
                "agent_id": agent.agent_id,
                "name": agent.name
            }
        except Exception as e:
            # On failure, transition to FAILED
            self.state.transition(agent.agent_id, AgentStatus.FAILED)
            self.state.set_error(agent.agent_id, f"Start failed: {e}")
            raise

    async def remove_agent(self, identifier: str) -> dict:
        """Remove an agent completely (destroys container and memory)"""
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Transition to REMOVING (valid from most states)
        self.state.transition(agent.agent_id, AgentStatus.REMOVING)

        # Get provisioned resources for cleanup
        resources = self.state.get_provisioned_resources(agent.agent_id)

        # Delete Matrix account (before removing from database)
        if self.matrix_manager:
            try:
                await self.matrix_manager.delete_agent_account(agent.agent_id)
                print(f"[REMOVE] Deleted Matrix account for agent {agent.agent_id}", flush=True)
            except Exception as e:
                print(f"[REMOVE] Warning: Failed to delete Matrix account for {agent.agent_id}: {e}", flush=True)
                # Continue with removal anyway

        # Remove container (will stop if running)
        if agent.container_id:
            try:
                self.docker.remove_container(agent.container_id)
            except Exception as e:
                print(f"[REMOVE] Warning: Failed to remove container: {e}", flush=True)

        # Remove memory directory if tracked
        if resources.get("memory_dir"):
            try:
                memory_dir = Path(resources["memory_dir"])
                if memory_dir.exists():
                    shutil.rmtree(memory_dir)
                    print(f"[REMOVE] Removed memory directory", flush=True)
            except Exception as e:
                print(f"[REMOVE] Warning: Failed to remove memory directory: {e}", flush=True)

        # Delete from state database (terminal operation - no transition possible after delete)
        self.state.delete_agent(agent.agent_id)

        return {"status": "removed", "agent_id": agent.agent_id}

    async def upgrade_agent(
        self,
        identifier: str,
        rebuild: bool = False,
        refresh_config: bool = False
    ) -> dict:
        """Upgrade an agent's container while preserving identity.

        Replaces the container with a new one using the same identity
        (memory, Matrix account, name). Optionally rebuilds the harness
        image and/or refreshes config from template.

        Args:
            identifier: Agent ID or name
            rebuild: If True, rebuild harness image before creating container
            refresh_config: If True, re-read config from template

        Returns:
            Dict with status, agent_id, name

        Raises:
            ValueError: Agent not found
            InvalidTransition: Agent in invalid state for upgrade
        """
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Validate current state allows upgrade
        current_status = AgentStatus(agent.status)
        if current_status not in [AgentStatus.RUNNING, AgentStatus.STOPPED, AgentStatus.FAILED]:
            raise InvalidTransition(
                f"Cannot upgrade agent in {current_status.value} state. "
                f"Agent must be running, stopped, or failed."
            )

        old_container_id = agent.container_id

        # Transition to UPGRADING
        self.state.transition(agent.agent_id, AgentStatus.UPGRADING)

        try:
            # Step 1: Stop container if running
            if current_status == AgentStatus.RUNNING:
                print(f"[UPGRADE] Stopping container...", flush=True)
                self.docker.stop_container(old_container_id)

            # Step 2: Remove old container
            print(f"[UPGRADE] Removing old container...", flush=True)
            self.docker.remove_container(old_container_id)

            # Step 3: Optionally rebuild image
            if rebuild:
                print(f"[UPGRADE] Rebuilding harness image...", flush=True)
                template = self.config.agents.get(agent.template)
                if template:
                    self.docker._build_image(template.image)

            # Step 4: Optionally refresh config from template
            if refresh_config:
                print(f"[UPGRADE] Refreshing config from template...", flush=True)
                template = self.config.agents.get(agent.template)
                if template:
                    self._refresh_agent_config(agent.agent_id, template)
                    # Reload agent to get updated config
                    agent = self.state.get_agent(agent.agent_id)

            # Step 5: Create new container with same identity
            print(f"[UPGRADE] Creating new container...", flush=True)
            template = self.config.agents.get(agent.template)
            environment = self._build_environment(agent.agent_id, template)
            new_host_url = self._start_container(agent.agent_id, template, environment)

            # Update host URL in config
            agent_config = agent.config.copy()
            agent_config["host_url"] = new_host_url
            self.state.update_agent_config(agent.agent_id, agent_config)

            # Step 6: Wait for health check
            print(f"[UPGRADE] Waiting for health check...", flush=True)
            updated_agent = self.state.get_agent(agent.agent_id)
            if not await self._wait_for_health(updated_agent.container_id, new_host_url):
                raise Exception("Container failed health check after upgrade")

            # Step 7: Transition to RUNNING
            self.state.transition(agent.agent_id, AgentStatus.RUNNING)

            print(f"[UPGRADE] Agent '{agent.name}' ({agent.agent_id}) upgraded successfully", flush=True)

            return {
                "status": "upgraded",
                "agent_id": agent.agent_id,
                "name": agent.name
            }

        except Exception as e:
            # On failure, transition to FAILED
            print(f"[UPGRADE] Upgrade failed: {e}", flush=True)
            self.state.transition(agent.agent_id, AgentStatus.FAILED)
            self.state.set_error(agent.agent_id, f"Upgrade failed: {e}")
            raise

    def _refresh_agent_config(self, agent_id: str, template: AgentTemplate):
        """Refresh agent config from template.

        Replaces stored config with fresh read from template.
        agent_id and name are preserved (stored in separate columns).
        """
        # Resolve prompt text
        prompt_text = None
        if template.prompt.text:
            prompt_text = template.prompt.text
        elif template.prompt.file:
            prompt_file = Path.home() / ".blue-fairy" / template.prompt.file
            if prompt_file.exists():
                prompt_text = prompt_file.read_text()

        # Build fresh config
        config = {
            "template": template.model,  # Keep for reference
            "model": template.model,
            "image": template.image,
            "prompt_text": prompt_text,
            "memory_enabled": template.memory.enabled if template.memory else False,
            "memory_backend": template.memory.backend if template.memory else "mem0",
            "remem_enabled": template.memory.remem_enabled if template.memory else False,
            "action_quota_limit": template.action_quota.limit if template.action_quota else None,
            "action_quota_window": template.action_quota.window_seconds if template.action_quota else None,
            "action_quota_low_threshold": template.action_quota.low_threshold if template.action_quota else 0.3,
            "subroutines_enabled": template.subroutines.enabled if template.subroutines else False,
            "subroutines_config": template.subroutines.model_dump() if template.subroutines else None,
            "mcp_servers": {name: server.model_dump() for name, server in template.mcp_servers.items()} if template.mcp_servers else None,
            "plugins": [p.model_dump() for p in template.plugins] if template.plugins else []
        }

        # Update stored config
        self.state.update_agent_config(agent_id, config)

        # Update observability settings
        observability_config_json = json.dumps({
            "events": {
                "decisions": template.observability.events.decisions,
                "memory_retrievals": template.observability.events.memory_retrievals,
                "communications": template.observability.events.communications,
                "reasoning": template.observability.events.reasoning
            }
        })

        # Update observability in database
        with self.state._get_connection() as conn:
            conn.execute("""
                UPDATE agents
                SET observability_level = ?,
                    self_reflection_enabled = ?,
                    observability_config = ?
                WHERE agent_id = ?
            """, (
                template.observability.level,
                template.observability.self_reflection_enabled,
                observability_config_json,
                agent_id
            ))

    def send_message(self, identifier: str, message: str) -> dict:
        """Send message to agent"""
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Get agent host URL from config (localhost:port for Docker Desktop)
        host_url = agent.config.get('host_url') or agent.config.get('ip_address')  # Fallback to old format
        if not host_url:
            raise Exception(f"Agent {identifier} has no host URL in config")

        # Call agent's /message endpoint
        url = f"http://{host_url}/message"

        try:
            response = requests.post(
                url,
                json={"content": message},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to send message: {e}")

    def _generate_agent_id(self, template_name: str) -> str:
        """Generate unique agent ID"""
        import random
        import string
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{template_name}-{suffix}"

    def _parse_cpu_quota(self, cpu_str: str) -> int:
        """Parse CPU string to quota (e.g., '0.5' -> 50000)"""
        cpu_float = float(cpu_str)
        return int(cpu_float * 100000)

    def _convert_memory_format(self, memory_str: str) -> str:
        """Convert Kubernetes-style memory to Docker format

        Examples:
        - 512Mi -> 512m
        - 1Gi -> 1g
        - 512m -> 512m (passthrough)
        """
        if not memory_str:
            return memory_str

        # Convert Ki, Mi, Gi to k, m, g
        memory_str = memory_str.replace('Ki', 'k')
        memory_str = memory_str.replace('Mi', 'm')
        memory_str = memory_str.replace('Gi', 'g')

        return memory_str

    async def _wait_for_health(self, container_id: str, agent_url: str, timeout: int = 120) -> bool:
        """Wait for container HTTP server to be ready

        Args:
            container_id: Docker container ID
            agent_url: Agent URL (e.g., "localhost:32768")
            timeout: Maximum seconds to wait

        Returns:
            True if HTTP server responds, False if timeout
        """
        import asyncio

        url = f"http://{agent_url}/health"
        print(f"[SPAWN] Health check URL: {url}", flush=True)

        for i in range(timeout * 2):
            # First check container is running
            container_status = self.docker.get_container_status(container_id)
            if container_status != "running":
                if i % 20 == 0:  # Log every 10 seconds
                    print(f"[SPAWN] Container status: {container_status}", flush=True)
                await asyncio.sleep(0.5)
                continue

            # Then check if HTTP server is responding
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print(f"[SPAWN] Health check passed after {i * 0.5}s", flush=True)
                    return True
                else:
                    if i % 20 == 0:
                        print(f"[SPAWN] Health check returned {response.status_code}", flush=True)
            except requests.exceptions.RequestException as e:
                if i % 20 == 0:  # Log every 10 seconds
                    print(f"[SPAWN] Health check failed: {e}", flush=True)

            await asyncio.sleep(0.5)

        print(f"[SPAWN] Health check timed out after {timeout}s", flush=True)
        return False

    def _initialize_agent(self, agent_id: str, host_url: str) -> str:
        """Initialize agent and get chosen name"""
        url = f"http://{host_url}/initialize"

        # Get existing names
        existing_names = self.state.get_existing_names()

        try:
            response = requests.post(
                url,
                json={"existing_names": existing_names},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            name = data["name"]
            public_key = data.get("public_key")

            # Update state with name and public key
            self.state.update_agent_name(agent_id, name)
            if public_key:
                self.state.update_agent_public_key(agent_id, public_key)

            return name
        except Exception as e:
            # Naming failed, use fallback
            print(f"[SPAWN] WARNING: Agent initialization failed: {e}", flush=True)
            print(f"[SPAWN] Using fallback name for {agent_id}", flush=True)
            fallback_name = f"Agent-{agent_id}"
            self.state.update_agent_name(agent_id, fallback_name)
            return fallback_name

    def _setup_git_access(self, agent_id: str) -> tuple[str, int]:
        """Set up git access for an agent.

        Generates SSH keypair, registers deploy key with GitHub,
        creates agent branch.

        Args:
            agent_id: Agent identifier

        Returns:
            Tuple of (private_key, github_key_id)

        Raises:
            ValueError if deploy_key_manager not configured
        """
        if not self.deploy_key_manager:
            raise ValueError("Deploy key manager not configured")

        # Generate keypair
        private_key, public_key = generate_ssh_keypair(agent_id)

        # Register with GitHub
        github_key_id = self.deploy_key_manager.register_deploy_key(
            agent_id=agent_id,
            public_key=public_key
        )

        # Create agent branch
        self.deploy_key_manager.create_agent_branch(agent_id)

        # Store in database
        self.state.create_deploy_key_record(
            agent_id=agent_id,
            github_key_id=github_key_id,
            public_key=public_key
        )

        return private_key, github_key_id

    def _create_pending_agent(self, agent_id: str, template_name: str, template: AgentTemplate):
        """Create agent record in PENDING state with complete config.

        This stores all configuration needed by harness to fetch via API,
        including resolved prompt_text, memory settings, etc.
        """
        # Resolve prompt text (either inline or from file)
        prompt_text = None
        if template.prompt.text:
            prompt_text = template.prompt.text
        elif template.prompt.file:
            # Resolve file path relative to config directory
            prompt_file = Path.home() / ".blue-fairy" / template.prompt.file
            if prompt_file.exists():
                prompt_text = prompt_file.read_text()
            else:
                raise ValueError(f"Prompt file not found: {prompt_file}")

        # Build complete config for harness
        config = {
            "template": template_name,
            "model": template.model,
            "image": template.image,
            "prompt_text": prompt_text,
            "memory_enabled": template.memory.enabled if template.memory else False,
            "memory_backend": template.memory.backend if template.memory else "mem0",
            "remem_enabled": template.memory.remem_enabled if template.memory else False,
            # Action quota settings (None values = unlimited)
            "action_quota_limit": template.action_quota.limit if template.action_quota else None,
            "action_quota_window": template.action_quota.window_seconds if template.action_quota else None,
            "action_quota_low_threshold": template.action_quota.low_threshold if template.action_quota else 0.3,
            # Subroutines settings (sleep, dreaming, memory consolidation)
            "subroutines_enabled": template.subroutines.enabled if template.subroutines else False,
            "subroutines_config": template.subroutines.model_dump() if template.subroutines else None,
            # MCP servers configuration
            "mcp_servers": {name: server.model_dump() for name, server in template.mcp_servers.items()} if template.mcp_servers else None,
            # Plugin configurations
            "plugins": [p.model_dump() for p in template.plugins] if template.plugins else []
        }

        # Extract observability configuration
        observability_level = template.observability.level
        self_reflection_enabled = template.observability.self_reflection_enabled
        observability_config_json = json.dumps({
            "events": {
                "decisions": template.observability.events.decisions,
                "memory_retrievals": template.observability.events.memory_retrievals,
                "communications": template.observability.events.communications,
                "reasoning": template.observability.events.reasoning
            }
        })

        # Create agent record in PENDING state
        self.state.create_agent(
            agent_id=agent_id,
            template=template_name,
            container_id="",  # No container yet
            config=config,
            observability_level=observability_level,
            self_reflection_enabled=self_reflection_enabled,
            observability_config=observability_config_json,
            status=AgentStatus.PENDING.value
        )

    def _provision_resources(self, agent_id: str, template: AgentTemplate) -> dict:
        """Provision resources for agent (memory dir, Matrix account).

        Returns dict of provisioned resources for rollback tracking.
        """
        resources = {}

        # Ensure Docker resources
        self.docker.ensure_network()
        self.docker.ensure_image(template.image)

        # Create memory directory if memory is enabled
        if template.memory and template.memory.enabled:
            memory_dir = Path.home() / ".blue-fairy" / "memory" / agent_id
            memory_dir.mkdir(parents=True, exist_ok=True)
            resources["memory_dir"] = str(memory_dir)

        # Create Matrix account if matrix_manager available
        if self.matrix_manager and self.matrix_manager.homeserver_url:
            # Matrix account creation is async
            resources["matrix_account"] = True

        return resources

    async def _provision_matrix_account(self, agent_id: str):
        """Provision Matrix account for agent (async operation)."""
        if not self.matrix_manager:
            return

        # Wait for Matrix to be fully initialized
        import asyncio
        max_wait = 60
        for i in range(max_wait):
            if self.matrix_manager.homeserver_url and self.matrix_manager.registration_secret:
                break
            if i == 0:
                print(f"[SPAWN] Waiting for Matrix to be fully initialized...", flush=True)
            await asyncio.sleep(1)

        if not self.matrix_manager.homeserver_url or not self.matrix_manager.registration_secret:
            print(f"[SPAWN] Warning: Matrix not ready after {max_wait}s, skipping account creation", flush=True)
            return

        try:
            await self.matrix_manager.create_agent_account(agent_id)
            print(f"[SPAWN] Created Matrix account for agent {agent_id}", flush=True)
        except Exception as e:
            print(f"[SPAWN] Warning: Failed to create Matrix account: {e}", flush=True)
            raise

    def _build_environment(self, agent_id: str, template: AgentTemplate) -> dict:
        """Build environment variables for container."""
        api_key = load_api_key()
        environment = {
            'AGENT_ID': agent_id,  # NEW - harness needs this to fetch config
            'ANTHROPIC_API_KEY': api_key,
            'OPENAI_API_KEY': 'sk-dummy-key-for-mem0-initialization',
            'SUPERVISOR_URL': 'http://host.docker.internal:8765',
            'OBSERVABILITY_LEVEL': str(template.observability.level)
        }

        # Serialize observability event preferences
        observability_config_json = json.dumps({
            "events": {
                "decisions": template.observability.events.decisions,
                "memory_retrievals": template.observability.events.memory_retrievals,
                "communications": template.observability.events.communications,
                "reasoning": template.observability.events.reasoning
            }
        })
        environment['OBSERVABILITY_CONFIG'] = observability_config_json

        # Add Matrix credentials if agent account was created
        if self.matrix_manager and self.matrix_manager.homeserver_url:
            matrix_account = self.state.get_agent_matrix_credentials(agent_id)
            if matrix_account:
                environment['MATRIX_USER_ID'] = matrix_account['matrix_user_id']
                environment['MATRIX_ACCESS_TOKEN'] = matrix_account['access_token']
                environment['MATRIX_HOMESERVER_URL'] = f"http://{MATRIX_CONTAINER_SERVICE_NAME}"
                print(f"[SPAWN] Added Matrix credentials to environment", flush=True)

        return environment

    def _start_container(self, agent_id: str, template: AgentTemplate, environment: dict) -> str:
        """Start Docker container and return host URL."""
        # Get memory directory if needed
        memory_dir = None
        if template.memory and template.memory.enabled:
            memory_dir = Path.home() / ".blue-fairy" / "memory" / agent_id

        # Parse resource limits
        memory = self._convert_memory_format(template.resources.memory) if template.resources and template.resources.memory else None
        cpu_quota = self._parse_cpu_quota(template.resources.cpu) if template.resources and template.resources.cpu else None

        # Run container
        print(f"[SPAWN] Starting container for agent {agent_id}", flush=True)
        container = self.docker.run_container(
            image=template.image,
            name=agent_id,
            memory=memory,
            cpu_quota=cpu_quota,
            environment=environment,
            memory_dir=memory_dir
        )

        # Update container ID in database
        self.state.update_agent_container(agent_id, container.id)

        # Get published port for host access
        time.sleep(1)
        container.reload()
        port_bindings = container.attrs['NetworkSettings']['Ports'].get('8080/tcp')
        if not port_bindings:
            raise Exception("Port 8080 not published")
        host_port = port_bindings[0]['HostPort']

        return f"localhost:{host_port}"

    async def _rollback(self, agent_id: str, error_message: str):
        """Clean up resources on spawn failure."""
        print(f"[SPAWN] Rolling back agent {agent_id}: {error_message}", flush=True)

        # Get provisioned resources
        resources = self.state.get_provisioned_resources(agent_id)

        # Get agent to check for container
        agent = self.state.get_agent(agent_id)

        # Remove container if created
        if agent and agent.container_id:
            try:
                self.docker.remove_container(agent.container_id)
                print(f"[SPAWN] Removed container {agent.container_id}", flush=True)
            except Exception as e:
                print(f"[SPAWN] Warning: Failed to remove container: {e}", flush=True)

        # Delete Matrix account if created
        if resources.get("matrix_account") and self.matrix_manager:
            try:
                await self.matrix_manager.delete_agent_account(agent_id)
                print(f"[SPAWN] Deleted Matrix account", flush=True)
            except Exception as e:
                print(f"[SPAWN] Warning: Failed to delete Matrix account: {e}", flush=True)

        # Remove memory directory if created
        if "memory_dir" in resources:
            try:
                memory_dir = Path(resources["memory_dir"])
                if memory_dir.exists():
                    shutil.rmtree(memory_dir)
                    print(f"[SPAWN] Removed memory directory", flush=True)
            except Exception as e:
                print(f"[SPAWN] Warning: Failed to remove memory directory: {e}", flush=True)

        # Set status to FAILED with error message
        self.state.transition(agent_id, AgentStatus.FAILED)
        self.state.set_error(agent_id, error_message)

    async def upgrade_agent_from_branch(
        self,
        identifier: str,
        commit_sha: str
    ) -> dict:
        """Upgrade agent to a specific commit from their branch.

        Full workflow with continuity testing:
        1. Build new image from agent's branch
        2. Plant continuity secret in agent's memory
        3. Create upgrade record
        4. Stop old container
        5. Start new container with new image
        6. Wait for health check
        7. Verify continuity (agent recalls secret)
        8. On success: tag image as passed, transition to RUNNING
        9. On failure: rollback to previous image, send failure message

        Args:
            identifier: Agent ID or name
            commit_sha: Git commit SHA to build from agent's branch

        Returns:
            Dict with status, agent_id, name
            - On success: {"status": "upgraded", "agent_id": ..., "name": ...}
            - On rollback: {"status": "rollback", "agent_id": ..., "reason": ...}

        Raises:
            ValueError: Agent not found
            InvalidTransition: Agent in invalid state for upgrade
        """
        agent = self.state.get_agent(identifier)
        if not agent:
            raise ValueError(f"Agent not found: {identifier}")

        # Validate current state allows upgrade
        current_status = AgentStatus(agent.status)
        if current_status not in [AgentStatus.RUNNING, AgentStatus.STOPPED, AgentStatus.FAILED]:
            raise InvalidTransition(
                f"Cannot upgrade agent in {current_status.value} state. "
                f"Agent must be running, stopped, or failed."
            )

        # Get memory backend for continuity planting
        memory_backend = agent.config.get("memory_backend", "mem0")
        old_container_id = agent.container_id
        old_image = agent.config.get("image", "blue-fairy/agent-base:latest")

        # Transition to UPGRADING
        self.state.transition(agent.agent_id, AgentStatus.UPGRADING)

        try:
            # Step 1: Determine version number
            history = self.state.get_agent_upgrade_history(agent.agent_id)
            version = len(history) + 1

            # Step 2: Build new image from agent's branch
            print(f"[UPGRADE] Building image from branch for agent {agent.agent_id} at commit {commit_sha}", flush=True)
            new_image_tag = self.docker.build_agent_image(agent.agent_id, commit_sha, version)

            # Step 3: Generate and plant continuity secret
            print(f"[UPGRADE] Planting continuity secret", flush=True)
            continuity_secret = generate_continuity_secret()
            plant_continuity_secret(
                agent_id=agent.agent_id,
                secret=continuity_secret,
                memory_backend=memory_backend,
                qdrant_url="http://localhost:6333"
            )

            # Step 4: Create upgrade record
            upgrade_id = self.state.create_upgrade_record(
                agent_id=agent.agent_id,
                commit_sha=commit_sha,
                image_tag=new_image_tag,
                previous_image_tag=old_image,
                continuity_secret=continuity_secret
            )
            self.state.update_upgrade_status(upgrade_id, "in_progress")

            # Step 5: Stop old container if running
            if current_status == AgentStatus.RUNNING:
                print(f"[UPGRADE] Stopping old container", flush=True)
                self.docker.stop_container(old_container_id)

            # Step 6: Remove old container
            print(f"[UPGRADE] Removing old container", flush=True)
            self.docker.remove_container(old_container_id)

            # Step 7: Start new container with new image
            print(f"[UPGRADE] Starting new container with image {new_image_tag}", flush=True)
            template = self.config.agents.get(agent.template)
            environment = self._build_environment(agent.agent_id, template)

            # Override image in config temporarily for this container
            agent_config = agent.config.copy()
            agent_config["image"] = new_image_tag

            # Get memory directory if enabled
            memory_dir = None
            if agent.config.get("memory_enabled"):
                memory_dir = Path.home() / ".blue-fairy" / "memory" / agent.agent_id

            # Parse resource limits
            memory = self._convert_memory_format(template.resources.memory) if template.resources and template.resources.memory else None
            cpu_quota = self._parse_cpu_quota(template.resources.cpu) if template.resources and template.resources.cpu else None

            # Run container with new image
            container = self.docker.run_container(
                image=new_image_tag,
                name=agent.agent_id,
                memory=memory,
                cpu_quota=cpu_quota,
                environment=environment,
                memory_dir=memory_dir
            )

            # Update container ID in database
            self.state.update_agent_container(agent.agent_id, container.id)

            # Get published port for host access
            time.sleep(1)
            container.reload()
            port_bindings = container.attrs['NetworkSettings']['Ports'].get('8080/tcp')
            if not port_bindings:
                raise Exception("Port 8080 not published")
            host_port = port_bindings[0]['HostPort']
            new_host_url = f"localhost:{host_port}"

            # Update host URL in config
            agent_config["host_url"] = new_host_url
            self.state.update_agent_config(agent.agent_id, agent_config)

            # Step 8: Wait for health check
            print(f"[UPGRADE] Waiting for health check", flush=True)
            if not await self._wait_for_health(container.id, new_host_url):
                raise Exception("Container failed health check after upgrade")

            # Step 9: Verify continuity
            print(f"[UPGRADE] Verifying continuity", flush=True)
            agent_url = f"http://{new_host_url}"
            continuity_passed = await verify_continuity(agent_url, continuity_secret)

            if not continuity_passed:
                # Continuity failed - trigger rollback
                print(f"[UPGRADE] Continuity verification FAILED - rolling back", flush=True)
                self.state.update_upgrade_status(
                    upgrade_id,
                    "failed",
                    failure_reason="Continuity verification failed"
                )
                self.docker.tag_image_status(new_image_tag, "failed")

                # Rollback to old image
                await self._rollback_upgrade(
                    agent=agent,
                    old_container_id=old_container_id,
                    old_image=old_image,
                    new_container_id=container.id,
                    template=template
                )

                return {
                    "status": "rollback",
                    "agent_id": agent.agent_id,
                    "reason": "continuity_failed"
                }

            # Step 10: Success - tag image as passed and transition to RUNNING
            print(f"[UPGRADE] Continuity verification PASSED", flush=True)
            self.state.update_upgrade_status(upgrade_id, "passed")
            self.docker.tag_image_status(new_image_tag, "passed")

            self.state.transition(agent.agent_id, AgentStatus.RUNNING)

            print(f"[UPGRADE] Agent '{agent.name}' ({agent.agent_id}) upgraded successfully", flush=True)

            return {
                "status": "upgraded",
                "agent_id": agent.agent_id,
                "name": agent.name
            }

        except Exception as e:
            # On failure, transition to FAILED
            print(f"[UPGRADE] Upgrade failed: {e}", flush=True)
            self.state.transition(agent.agent_id, AgentStatus.FAILED)
            self.state.set_error(agent.agent_id, f"Upgrade from branch failed: {e}")
            raise

    async def _rollback_upgrade(
        self,
        agent,
        old_container_id: str,
        old_image: str,
        new_container_id: str,
        template: AgentTemplate
    ):
        """Rollback upgrade by restoring old container.

        Args:
            agent: Agent record from database
            old_container_id: Container ID of previous version
            old_image: Docker image tag of previous version
            new_container_id: Container ID of failed new version
            template: Agent template for resource limits
        """
        print(f"[ROLLBACK] Rolling back agent {agent.agent_id}", flush=True)

        try:
            # Stop and remove new (failed) container
            print(f"[ROLLBACK] Stopping new container", flush=True)
            self.docker.stop_container(new_container_id)

            print(f"[ROLLBACK] Removing new container", flush=True)
            self.docker.remove_container(new_container_id)

            # Recreate old container
            print(f"[ROLLBACK] Recreating old container with image {old_image}", flush=True)
            environment = self._build_environment(agent.agent_id, template)

            # Get memory directory if enabled
            memory_dir = None
            if agent.config.get("memory_enabled"):
                memory_dir = Path.home() / ".blue-fairy" / "memory" / agent.agent_id

            # Parse resource limits
            memory = self._convert_memory_format(template.resources.memory) if template.resources and template.resources.memory else None
            cpu_quota = self._parse_cpu_quota(template.resources.cpu) if template.resources and template.resources.cpu else None

            # Run container with old image
            container = self.docker.run_container(
                image=old_image,
                name=agent.agent_id,
                memory=memory,
                cpu_quota=cpu_quota,
                environment=environment,
                memory_dir=memory_dir
            )

            # Update container ID
            self.state.update_agent_container(agent.agent_id, container.id)

            # Get published port
            time.sleep(1)
            container.reload()
            port_bindings = container.attrs['NetworkSettings']['Ports'].get('8080/tcp')
            if port_bindings:
                host_port = port_bindings[0]['HostPort']
                host_url = f"localhost:{host_port}"
            else:
                # Fallback to IP-based approach
                host_url = container.attrs['NetworkSettings']['Networks']['blue-fairy-net']['IPAddress'] + ":8080"

            # Update host URL
            agent_config = agent.config.copy()
            agent_config["host_url"] = host_url
            agent_config["image"] = old_image  # Restore old image
            self.state.update_agent_config(agent.agent_id, agent_config)

            # Wait for health check
            print(f"[ROLLBACK] Waiting for health check", flush=True)
            if not await self._wait_for_health(container.id, host_url):
                raise Exception("Rollback container failed health check")

            # Transition to RUNNING
            self.state.transition(agent.agent_id, AgentStatus.RUNNING)

            print(f"[ROLLBACK] Rollback complete - agent restored to previous version", flush=True)

        except Exception as e:
            print(f"[ROLLBACK] Rollback failed: {e}", flush=True)
            self.state.transition(agent.agent_id, AgentStatus.FAILED)
            self.state.set_error(agent.agent_id, f"Rollback failed: {e}")
            raise
